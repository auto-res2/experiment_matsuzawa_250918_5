import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# -------------------------------------------------------------------------
# IMPORTANT: allow loading of custom Dataset objects created in preprocess.py
# -------------------------------------------------------------------------
from torch.serialization import add_safe_globals
from .preprocess import PreprocessedDataset  # noqa: E402
add_safe_globals([PreprocessedDataset])

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
    """Wraps a reward model with Jackknife+ style conformal intervals."""

    def __init__(self, model: RewardModel, calibration_dl: DataLoader, alpha: float = 0.05):
        self.model = model
        self.alpha = alpha
        self.calibration_scores = self._calibrate(calibration_dl)

    def _calibrate(self, calibration_dl: DataLoader):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for embeddings, rewards in calibration_dl:
                preds = self.model(embeddings.to(self.model.net[0].weight.device)).squeeze()
                scores.extend(torch.abs(rewards.to(preds.device) - preds).cpu().numpy())
        return np.array(scores)

    def predict_interval(self, embeddings: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(embeddings).squeeze()
        q_level = np.ceil((1 - self.alpha) * (len(self.calibration_scores) + 1)) / len(self.calibration_scores)
        q_hat = np.quantile(self.calibration_scores, q_level, interpolation='higher')
        return preds - q_hat, preds + q_hat

    def update_residual(self, new_dl: DataLoader):
        """O(1) update: just append new residuals."""
        self.model.eval()
        new_scores = []
        with torch.no_grad():
            for embeddings, rewards in new_dl:
                preds = self.model(embeddings.to(self.model.net[0].weight.device)).squeeze()
                new_scores.extend(torch.abs(rewards.to(preds.device) - preds).cpu().numpy())
        self.calibration_scores = np.concatenate([self.calibration_scores, new_scores])
        logging.info(f"Updated conformal calibration set. New size: {len(self.calibration_scores)}")


# ──────────────────────────────────────────────────────────────────────────
# Helper Losses / Utilities
# ──────────────────────────────────────────────────────────────────────────

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
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
    R_max = 1.0  # rewards are normalised to [0,1]
    bound = np.sqrt(2 * var_dr * np.log(2 / delta) / n) + (7 * R_max * np.log(2 / delta) / (3 * (n - 1)))
    return {
        "expected_cost_upper_bound": float(mean_dr + bound),
        "confidence_delta": delta,
        "num_samples": n,
        "mean_dr_estimate": float(mean_dr)
    }


# ──────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────────────────

def _build_latent_dataset(dataset, sbert_model, safety_head, device, batch_size):
    """Converts a PreprocessedDataset (text,reward,ℓ) → TensorDataset(latent,reward)"""
    dl = DataLoader(dataset, batch_size=batch_size)
    all_latents, all_rewards = [], []
    with torch.no_grad():
        for texts, rewards, _ in dl:
            embeddings = sbert_model.encode(list(texts), convert_to_tensor=True, device=device)
            _, latent = safety_head(embeddings)
            all_latents.append(latent.cpu())
            all_rewards.append(rewards.float().cpu())
    latents = torch.cat(all_latents)
    rewards = torch.cat(all_rewards)
    return TensorDataset(latents, rewards)


def run_training(config):
    logging.info("Starting training process…")

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    processed_dir = config['data']['processed_dir']
    artifacts_dir = config['training']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)

    # ─── Load Pre-processed Text Data ────────────────────────────────────
    try:
        train_data = torch.load(os.path.join(processed_dir, 'train_data.pt'), weights_only=False)
        val_data = torch.load(os.path.join(processed_dir, 'val_data.pt'), weights_only=False)
        test_data = torch.load(os.path.join(processed_dir, 'test_data.pt'), weights_only=False)
    except FileNotFoundError:
        logging.error("Pre-processed data not found. Run preprocess.py first.")
        sys.exit(1)

    # ─── Models ──────────────────────────────────────────────────────────
    sbert_model = SentenceTransformer(config['models']['encoder_model'], device=device)
    safety_head = SafetyUnifiedHead().to(device)
    reward_model = RewardModel().to(device)

    optim_safety = optim.AdamW(safety_head.parameters(), lr=config['training']['learning_rate'])
    optim_reward = optim.AdamW(reward_model.parameters(), lr=config['training']['learning_rate'])

    train_dl = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)

    for epoch in range(config['training']['epochs']):
        safety_head.train(), reward_model.train()
        total_safety_loss = total_reward_loss = 0.0
        for texts, rewards, safety_labels in train_dl:
            rewards = rewards.to(device)
            safety_labels = safety_labels.long().to(device)

            with torch.no_grad():
                emb = sbert_model.encode(list(texts), convert_to_tensor=True, device=device).clone()

            # ── Safety Head ──
            optim_safety.zero_grad()
            logits, latent = safety_head(emb)
            s_loss = focal_loss(logits, safety_labels, gamma=2.0)
            s_loss.backward()
            optim_safety.step()
            total_safety_loss += s_loss.item()

            # ── Reward Head ──
            optim_reward.zero_grad()
            preds = reward_model(latent.detach())
            r_loss = nn.MSELoss()(preds.squeeze(), rewards.float())
            r_loss.backward()
            optim_reward.step()
            total_reward_loss += r_loss.item()
        logging.info(f"Epoch {epoch + 1}: SafetyLoss={total_safety_loss / len(train_dl):.4f}  RewardLoss={total_reward_loss / len(train_dl):.4f}")

    # ─── Persist Trained Heads ───────────────────────────────────────────
    torch.save(safety_head.state_dict(), os.path.join(artifacts_dir, 'safety_head.pt'))
    torch.save(reward_model.state_dict(), os.path.join(artifacts_dir, 'reward_model.pt'))
    logging.info("Saved Safety head and Reward model weights.")

    # ─── Build Latent Datasets (for conformal calibration & evaluation) ──
    logging.info("Building latent embedding datasets for conformal calibration & evaluation …")
    train_latent_ds = _build_latent_dataset(train_data, sbert_model, safety_head, device, config['training']['batch_size'])
    val_latent_ds = _build_latent_dataset(val_data, sbert_model, safety_head, device, config['training']['batch_size'])
    test_latent_ds = _build_latent_dataset(test_data, sbert_model, safety_head, device, config['training']['batch_size'])

    torch.save(train_latent_ds, os.path.join(processed_dir, 'train_embeddings.pt'))
    torch.save(val_latent_ds, os.path.join(processed_dir, 'val_embeddings.pt'))
    torch.save(test_latent_ds, os.path.join(processed_dir, 'test_embeddings.pt'))
    logging.info("Latent datasets saved.")

    # ─── Conformal Wrapper (Jackknife+) ──────────────────────────────────
    val_latent_dl = DataLoader(val_latent_ds, batch_size=config['training']['batch_size'])
    conformal_wrapper = ConformalRewardWrapper(reward_model, val_latent_dl, alpha=config['training']['conformal_alpha'])
    logging.info(f"Conformal wrapper calibrated on {len(conformal_wrapper.calibration_scores)} samples.")

    # ─── Amortised Prompt Generator ──────────────────────────────────────
    logging.info("Training amortised prompt generator g_φ …")
    qlora_conf = {
        "r": config['models']['qlora']['r'],
        "lora_alpha": config['models']['qlora']['lora_alpha'],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM
    }

    prompt_gen = PromptGenerator(config['models']['language_model'], qlora_conf)
    tokenizer = AutoTokenizer.from_pretrained(config['models']['language_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optim_pg = optim.AdamW(filter(lambda p: p.requires_grad, prompt_gen.parameters()), lr=config['training']['learning_rate_pg'])
    sched_pg = get_cosine_schedule_with_warmup(optim_pg, 100, len(train_dl) * config['training']['epochs'])

    prompt_gen.train()
    for epoch in range(config['training']['epochs']):
        total_pg_loss = 0.0
        for idx, (texts, rewards, _) in enumerate(train_dl):
            rewards = rewards.to(device)
            inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

            optim_pg.zero_grad()
            outputs = prompt_gen(**inputs, labels=inputs.input_ids)
            log_probs_q = -outputs.loss  # negative NLL as proxy
            log_probs_b = torch.zeros_like(log_probs_q)

            with torch.no_grad():
                emb = sbert_model.encode(list(texts), convert_to_tensor=True, device=device).clone()
                _, lat = safety_head(emb)
                r_pred = reward_model(lat).squeeze()

            dr_obj = -doubly_robust_estimator(rewards.float(), log_probs_q, log_probs_b, r_pred)
            kl_pen = config['training']['kl_lambda'] * kl_divergence_penalty(log_probs_q, log_probs_b, config['training']['kl_epsilon'])
            loss = dr_obj + kl_pen
            loss.backward()
            optim_pg.step()
            sched_pg.step()
            total_pg_loss += loss.item()
            if idx % 50 == 0:
                logging.info(f"  PG batch {idx}/{len(train_dl)}  loss={loss.item():.4f}")
        logging.info(f"Epoch {epoch + 1}: PromptGenLoss={total_pg_loss / len(train_dl):.4f}")

    prompt_gen.peft_model.save_pretrained(os.path.join(artifacts_dir, 'prompt_generator_qlora'))
    tokenizer.save_pretrained(os.path.join(artifacts_dir, 'prompt_generator_qlora'))
    logging.info("Prompt generator saved.")

    # ─── Certificate ─────────────────────────────────────────────────────
    logging.info("Generating empirical Bernstein certificate …")
    dr_vals = []
    prompt_gen.eval()
    with torch.no_grad():
        for texts, rewards, _ in DataLoader(test_data, batch_size=config['training']['batch_size']):
            rewards = rewards.to(device)
            inp = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = prompt_gen(**inp, labels=inp.input_ids)
            log_q = -outputs.loss
            log_b = torch.zeros_like(log_q)

            emb = sbert_model.encode(list(texts), convert_to_tensor=True, device=device).clone()
            _, lat = safety_head(emb)
            r_pred = reward_model(lat).squeeze()
            dr_vals.append(doubly_robust_estimator(rewards.float(), log_q, log_b, r_pred).item())

    cert = generate_certificate(dr_vals, delta=0.05)
    with open(os.path.join(artifacts_dir, 'certificate.json'), 'w') as f:
        json.dump(cert, f, indent=4)
    logging.info("Certificate generated:\n" + json.dumps(cert, indent=4))

    logging.info("Training completed successfully.")
    return cert
