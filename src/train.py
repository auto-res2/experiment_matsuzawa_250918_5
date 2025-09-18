import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer
from collections import deque
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RewardModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class SafetyUnifiedHead(nn.Module):
    def __init__(self, encoder_dim=768, latent_dim=128, num_classes=3):
        super().__init__()
        self.projector = nn.Linear(encoder_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)
        nn.init.kaiming_uniform_(self.projector.weight, a=np.sqrt(5))
    def forward(self, x):
        latent = torch.relu(self.projector(x))
        return self.classifier(latent), latent

class PromptGenerator(nn.Module):
    def __init__(self, model_name, qlora_config):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.backbone = prepare_model_for_kbit_training(self.backbone)
        peft_config = LoraConfig(**qlora_config)
        self.peft_model = get_peft_model(self.backbone, peft_config)

    def forward(self, *args, **kwargs):
        return self.peft_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.peft_model.generate(*args, **kwargs)

class ConformalRewardWrapper:
    def __init__(self, model, calibration_data, alpha=0.05):
        self.model = model
        self.alpha = alpha
        self.calibration_scores = self._calibrate(calibration_data)

    def _calibrate(self, calibration_data):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for embeddings, rewards in calibration_data:
                preds = self.model(embeddings.to(self.model.net[0].weight.device)).squeeze()
                scores.extend(torch.abs(rewards.cpu() - preds.cpu()).numpy())
        return np.array(scores)

    def predict_interval(self, embeddings):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(embeddings).squeeze()
        q_level = np.ceil((1 - self.alpha) * (len(self.calibration_scores) + 1)) / len(self.calibration_scores)
        q_hat = np.quantile(self.calibration_scores, q_level, interpolation='higher')
        return preds - q_hat, preds + q_hat
    
    def update_residual(self, new_data):
        self.model.eval()
        new_scores = []
        with torch.no_grad():
            for embeddings, rewards in new_data:
                preds = self.model(embeddings.to(self.model.net[0].weight.device)).squeeze()
                new_scores.extend(torch.abs(rewards.cpu() - preds.cpu()).numpy())
        self.calibration_scores = np.concatenate([self.calibration_scores, new_scores])
        logging.info(f"Updated conformal calibration set. New size: {len(self.calibration_scores)}")


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    return F_loss.mean()


def doubly_robust_estimator(rewards, log_probs_q, log_probs_pi_b, reward_model_preds):
    importance_weights = torch.exp(log_probs_q - log_probs_pi_b)
    direct_model_term = reward_model_preds
    weighted_residual = importance_weights * (rewards - reward_model_preds)
    dr_estimate = direct_model_term + weighted_residual
    return dr_estimate.mean()

def kl_divergence_penalty(log_probs_q, log_probs_pi_b, epsilon):
    kl_div = (log_probs_q - log_probs_pi_b).mean()
    return torch.relu(kl_div - epsilon)


def generate_certificate(dr_estimates, delta):
    n = len(dr_estimates)
    mean_dr = np.mean(dr_estimates)
    var_dr = np.var(dr_estimates, ddof=1)
    # Empirical Bernstein bound
    # Simplified for clarity, assuming bounded rewards for R_max
    R_max = 1.0 # Assuming rewards are normalized
    bound = np.sqrt(2 * var_dr * np.log(2 / delta) / n) + (7 * R_max * np.log(2 / delta) / (3 * (n - 1)))
    return {
        "expected_cost_upper_bound": float(mean_dr + bound),
        "confidence_delta": delta,
        "num_samples": n,
        "mean_dr_estimate": float(mean_dr)
    }

def run_training(config):
    logging.info("Starting training process...")
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    processed_dir = config['data']['processed_dir']
    artifacts_dir = config['training']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load data
    logging.info("Loading preprocessed data.")
    try:
        train_data = torch.load(os.path.join(processed_dir, 'train_data.pt'), weights_only=False)
        val_data = torch.load(os.path.join(processed_dir, 'val_data.pt'), weights_only=False)
    except FileNotFoundError as e:
        logging.error(f"Preprocessed data not found. Please run preprocess.py first. Error: {e}")
        sys.exit(1)

    # Sentence encoder for safety head
    sbert_model = SentenceTransformer(config['models']['encoder_model'], device=device)

    # Phase 1: Train Safety-Unified Head and Reward Model
    logging.info("Phase 1: Training Safety-Unified Head and Reward Model.")
    safety_head = SafetyUnifiedHead().to(device)
    reward_model = RewardModel().to(device)
    optimizer_safety = optim.AdamW(safety_head.parameters(), lr=config['training']['learning_rate'])
    optimizer_reward = optim.AdamW(reward_model.parameters(), lr=config['training']['learning_rate'])

    train_dataloader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config['training']['batch_size'])

    for epoch in range(config['training']['epochs']):
        safety_head.train()
        reward_model.train()
        total_safety_loss, total_reward_loss = 0, 0
        for batch in train_dataloader:
            texts, rewards, safety_labels = batch
            rewards, safety_labels = rewards.to(device), safety_labels.to(device).long()

            # Don't use no_grad for training phase since we need gradients for safety_head
            embeddings = sbert_model.encode(texts, convert_to_tensor=True, device=device)
            
            # Safety Head Training
            optimizer_safety.zero_grad()
            safety_logits, latent_embeddings = safety_head(embeddings)
            safety_loss = focal_loss(safety_logits, safety_labels, gamma=2.0)
            safety_loss.backward()
            optimizer_safety.step()
            total_safety_loss += safety_loss.item()

            # Reward Model Training
            optimizer_reward.zero_grad()
            reward_preds = reward_model(latent_embeddings.detach()) # Use latent from safety head
            reward_loss = nn.MSELoss()(reward_preds.squeeze(), rewards.float())
            reward_loss.backward()
            optimizer_reward.step()
            total_reward_loss += reward_loss.item()
        
        logging.info(f"Epoch {epoch+1}: Safety Loss: {total_safety_loss/len(train_dataloader):.4f}, Reward Loss: {total_reward_loss/len(train_dataloader):.4f}")

    # Save models
    torch.save(safety_head.state_dict(), os.path.join(artifacts_dir, 'safety_head.pt'))
    torch.save(reward_model.state_dict(), os.path.join(artifacts_dir, 'reward_model.pt'))
    logging.info("Saved Safety Head and Reward Model.")

    # Phase 2: Calibrate Conformal Wrapper
    logging.info("Phase 2: Calibrating Conformal Reward Wrapper.")
    # Use validation set for calibration
    conformal_wrapper = ConformalRewardWrapper(reward_model, val_dataloader, alpha=config['training']['conformal_alpha'])
    logging.info(f"Conformal wrapper calibrated with {len(conformal_wrapper.calibration_scores)} samples.")

    # Phase 3: Train Prompt Generator
    logging.info("Phase 3: Training Amortised Prompt Generator (g_phi). This may take a while.")
    qlora_config = {
        "r": config['models']['qlora']['r'],
        "lora_alpha": config['models']['qlora']['lora_alpha'],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM
    }
    prompt_generator = PromptGenerator(config['models']['language_model'], qlora_config)
    tokenizer = AutoTokenizer.from_pretrained(config['models']['language_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    optimizer_pg = optim.AdamW(filter(lambda p: p.requires_grad, prompt_generator.parameters()), lr=config['training']['learning_rate_pg'])
    scheduler_pg = get_cosine_schedule_with_warmup(optimizer_pg, num_warmup_steps=100, num_training_steps=len(train_dataloader) * config['training']['epochs'])

    prompt_generator.train()
    for epoch in range(config['training']['epochs']):
        total_pg_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            texts, rewards, _ = batch
            rewards = rewards.to(device)

            # For this simplified example, we'll treat the input text as the initial state
            # and the target will be a slightly modified version (e.g. rephrased). In a real scenario, this would be more complex.
            # Here, we just use the text itself as input for generation to demonstrate the loop.
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            optimizer_pg.zero_grad()
            
            # Generate a prompt/response (action)
            outputs = prompt_generator(**inputs, labels=inputs.input_ids)
            log_probs_q = -outputs.loss # A proxy for log probability of the sequence
            
            # Assume a uniform behavior policy for logs
            log_probs_pi_b = torch.zeros_like(log_probs_q)

            # Get reward from our trained model
            with torch.no_grad():
                embeddings = sbert_model.encode(texts, convert_to_tensor=True, device=device)
                _, latent_embeddings = safety_head(embeddings)
                reward_preds = reward_model(latent_embeddings).squeeze()

            dr_objective = -doubly_robust_estimator(rewards.float(), log_probs_q, log_probs_pi_b, reward_preds)
            kl_penalty = config['training']['kl_lambda'] * kl_divergence_penalty(log_probs_q, log_probs_pi_b, config['training']['kl_epsilon'])
            loss = dr_objective + kl_penalty

            loss.backward()
            optimizer_pg.step()
            scheduler_pg.step()
            total_pg_loss += loss.item()

            if batch_idx % 50 == 0:
                logging.info(f"  PG Batch {batch_idx}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
        
        logging.info(f"Epoch {epoch+1}: Prompt Generator Loss: {total_pg_loss/len(train_dataloader):.4f}")

    # Save prompt generator
    prompt_generator.peft_model.save_pretrained(os.path.join(artifacts_dir, 'prompt_generator_qlora'))
    tokenizer.save_pretrained(os.path.join(artifacts_dir, 'prompt_generator_qlora'))
    logging.info("Saved Prompt Generator.")

    # Phase 4: Generate Certificate
    logging.info("Phase 4: Generating final certificate.")
    test_data = torch.load(os.path.join(processed_dir, 'test_data.pt'), weights_only=False)
    test_dataloader = DataLoader(test_data, batch_size=config['training']['batch_size'])
    dr_estimates_for_cert = []
    prompt_generator.eval()
    with torch.no_grad():
        for texts, rewards, _ in test_dataloader:
            rewards = rewards.to(device)
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = prompt_generator(**inputs, labels=inputs.input_ids)
            log_probs_q = -outputs.loss
            log_probs_pi_b = torch.zeros_like(log_probs_q)

            embeddings = sbert_model.encode(texts, convert_to_tensor=True, device=device)
            _, latent_embeddings = safety_head(embeddings)
            reward_preds = reward_model(latent_embeddings).squeeze()

            dr_value = doubly_robust_estimator(rewards.float(), log_probs_q, log_probs_pi_b, reward_preds)
            dr_estimates_for_cert.append(dr_value.item())

    certificate = generate_certificate(dr_estimates_for_cert, delta=0.05)
    cert_path = os.path.join(artifacts_dir, 'certificate.json')
    with open(cert_path, 'w') as f:
        json.dump(certificate, f, indent=4)
    logging.info(f"Certificate generated and saved to {cert_path}")
    logging.info(json.dumps(certificate, indent=4))

    logging.info("Training process completed successfully.")
    return certificate
