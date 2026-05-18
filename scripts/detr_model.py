# ============================================================
# DETR ARCHITECTURE AND TRAINING UTILITIES
# ============================================================

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


# ============================================================
# Spectrum Encoder
# ============================================================

class SpectrumEncoder(nn.Module):
    """
    1D CNN spectral feature extractor.

    Input:
        x: [B, 1, N_points]

    Output:
        memory: [B, L, d_model]
    """

    def __init__(self, d_model=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(128, d_model, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.net(x)              # [B, d_model, L]
        memory = h.transpose(1, 2)   # [B, L, d_model]
        return memory


# ============================================================
# Spectrum DETR
# ============================================================

class SpectrumDETR(nn.Module):
    """
    DETR-style set predictor for XPS spectra.

    Architecture:
        spectrum -> 1D CNN encoder -> Transformer encoder -> Transformer decoder
                 -> environment embeddings + confidence logits

    Outputs:
        env_preds:   [B, Q, d_env]
        conf_logits: [B, Q]
    """

    def __init__(
        self,
        d_model=256,
        d_env=128,
        n_queries=8,
        n_heads=8,
        n_layers=3,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = SpectrumEncoder(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
        )

        # Learnable environment queries
        self.query_embed = nn.Parameter(
            torch.randn(n_queries, d_model) * 0.02
        )

        # Prediction heads
        self.env_head = nn.Linear(d_model, d_env)
        self.conf_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: [B, 1, N_points]

        Returns:
            env_preds:   [B, Q, d_env]
            conf_logits: [B, Q]
        """

        B = x.size(0)

        # CNN feature extraction
        memory = self.encoder(x)                       # [B, L, d_model]

        # Transformer encoder for global spectral context
        memory = self.transformer_encoder(memory)      # [B, L, d_model]

        # Expand learned queries for batch
        queries = self.query_embed.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, d_model]

        # Transformer decoder set prediction
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
        )                                               # [B, Q, d_model]

        env_preds = self.env_head(decoded)              # [B, Q, d_env]
        conf_logits = self.conf_head(decoded).squeeze(-1)  # [B, Q]

        return env_preds, conf_logits


# ============================================================
# Hungarian Matching
# ============================================================

@torch.no_grad()
def hungarian_match(
    env_preds,
    conf_logits,
    targets,
    w_emb=1.0,
    w_conf=0.5,
    use_cosine=True,
):
    """
    Match predicted environment embeddings to target embeddings.

    Args:
        env_preds:   [B, Q, d_env]
        conf_logits: [B, Q]
        targets:     list of tensors, each [M_i, d_env]
        w_emb:       embedding-distance cost weight
        w_conf:      confidence cost weight
        use_cosine:  if True, use cosine distance; otherwise use Euclidean distance

    Returns:
        matches: list of (pred_idx, tgt_idx) tensors for each batch element
    """

    B, Q, _ = env_preds.shape
    matches = []

    for b in range(B):
        T = targets[b]        # [M, d_env]
        M = T.shape[0]

        if M == 0:
            matches.append((
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            ))
            continue

        P = env_preds[b]      # [Q, d_env]

        # Embedding cost: [Q, M]
        if use_cosine:
            Pn = F.normalize(P, dim=-1)
            Tn = F.normalize(T, dim=-1)
            cost_emb = 1.0 - (Pn @ Tn.T)
        else:
            cost_emb = torch.cdist(P, T, p=2)

        # Confidence cost: lower cost for high-confidence matched predictions
        cost_conf = -F.logsigmoid(conf_logits[b]).unsqueeze(1).expand(Q, M)

        cost = (w_emb * cost_emb + w_conf * cost_conf).detach().cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(cost)

        matches.append((
            torch.as_tensor(row_ind, dtype=torch.long),
            torch.as_tensor(col_ind, dtype=torch.long),
        ))

    return matches


# ============================================================
# Set Prediction Loss
# ============================================================

def detr_set_loss(
    env_preds,
    conf_logits,
    targets,
    matches,
    w_emb=1.0,
    w_conf=0.5,
    use_cosine=True,
):
    """
    DETR-style set loss.

    Matched predictions are supervised with an embedding loss and confidence target 1.
    Unmatched predictions are supervised with confidence target 0.

    Args:
        env_preds:   [B, Q, d_env]
        conf_logits: [B, Q]
        targets:     list of tensors, each [M_i, d_env]
        matches:     list of (pred_idx, tgt_idx) tensors
        w_emb:       embedding loss weight
        w_conf:      confidence loss weight
        use_cosine:  if True, use cosine distance; otherwise use MSE

    Returns:
        total_loss, parts
    """

    B, Q, _ = env_preds.shape

    total_emb = env_preds.new_tensor(0.0)
    total_conf = env_preds.new_tensor(0.0)

    for b in range(B):
        pred_idx, tgt_idx = matches[b]

        pred_idx = pred_idx.to(env_preds.device)
        tgt_idx = tgt_idx.to(targets[b].device)

        # Confidence targets: matched = 1, unmatched = 0
        conf_target = torch.zeros((Q,), device=conf_logits.device)

        if pred_idx.numel() > 0:
            conf_target[pred_idx] = 1.0

        total_conf = total_conf + F.binary_cross_entropy_with_logits(
            conf_logits[b],
            conf_target,
        )

        # Embedding loss is applied only to matched prediction-target pairs
        if pred_idx.numel() > 0:
            P = env_preds[b, pred_idx]      # [K, d_env]
            T = targets[b][tgt_idx]         # [K, d_env]

            if use_cosine:
                emb_loss = 1.0 - F.cosine_similarity(P, T, dim=-1)
                total_emb = total_emb + emb_loss.mean()
            else:
                total_emb = total_emb + F.mse_loss(P, T)

    total_emb = total_emb / B
    total_conf = total_conf / B

    total_loss = w_emb * total_emb + w_conf * total_conf

    return total_loss, {
        "emb": float(total_emb.item()),
        "conf": float(total_conf.item()),
    }


# ============================================================
# Evaluation Helper
# ============================================================

def evaluate_detr(
    model,
    loader,
    device,
    w_emb=1.0,
    w_conf=0.5,
    use_cosine=True,
):
    model.eval()

    running_loss = 0.0
    running_emb = 0.0
    running_conf = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["spectra"].to(device)
            targets = [t.to(device) for t in batch["targets"]]

            env_preds, conf_logits = model(x)

            matches = hungarian_match(
                env_preds,
                conf_logits,
                targets,
                w_emb=w_emb,
                w_conf=w_conf,
                use_cosine=use_cosine,
            )

            loss, parts = detr_set_loss(
                env_preds,
                conf_logits,
                targets,
                matches,
                w_emb=w_emb,
                w_conf=w_conf,
                use_cosine=use_cosine,
            )

            running_loss += float(loss.item())
            running_emb += float(parts["emb"])
            running_conf += float(parts["conf"])
            n_batches += 1

    return {
        "loss": running_loss / max(1, n_batches),
        "emb": running_emb / max(1, n_batches),
        "conf": running_conf / max(1, n_batches),
    }


# ============================================================
# Training and Validation Loop
# ============================================================

def train_detr(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=1e-4,
    w_emb=1.0,
    w_conf=0.5,
    use_cosine=True,
    ckpt_path=None,
    metrics_path=None,
    ood_loader=None,
):
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    history = {
        "loss": [],
        "emb": [],
        "conf": [],
        "val_loss": [],
        "val_emb": [],
        "val_conf": [],
    }

    if ood_loader is not None:
        history["ood_loss"] = []
        history["ood_emb"] = []
        history["ood_conf"] = []

    for ep in range(1, epochs + 1):
        # -------------------------
        # Training
        # -------------------------
        model.train()

        running_loss = 0.0
        running_emb = 0.0
        running_conf = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["spectra"].to(device)
            targets = [t.to(device) for t in batch["targets"]]

            env_preds, conf_logits = model(x)

            matches = hungarian_match(
                env_preds,
                conf_logits,
                targets,
                w_emb=w_emb,
                w_conf=w_conf,
                use_cosine=use_cosine,
            )

            loss, parts = detr_set_loss(
                env_preds,
                conf_logits,
                targets,
                matches,
                w_emb=w_emb,
                w_conf=w_conf,
                use_cosine=use_cosine,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += float(loss.item())
            running_emb += float(parts["emb"])
            running_conf += float(parts["conf"])
            n_batches += 1

        epoch_loss = running_loss / max(1, n_batches)
        epoch_emb = running_emb / max(1, n_batches)
        epoch_conf = running_conf / max(1, n_batches)

        history["loss"].append(epoch_loss)
        history["emb"].append(epoch_emb)
        history["conf"].append(epoch_conf)

        # -------------------------
        # Validation
        # -------------------------
        val_metrics = evaluate_detr(
            model,
            val_loader,
            device,
            w_emb=w_emb,
            w_conf=w_conf,
            use_cosine=use_cosine,
        )

        history["val_loss"].append(val_metrics["loss"])
        history["val_emb"].append(val_metrics["emb"])
        history["val_conf"].append(val_metrics["conf"])

        msg = (
            f"epoch {ep:02d} | "
            f"train loss {epoch_loss:.4f} | emb {epoch_emb:.4f} | conf {epoch_conf:.4f} || "
            f"val loss {val_metrics['loss']:.4f} | emb {val_metrics['emb']:.4f} | conf {val_metrics['conf']:.4f}"
        )

        # -------------------------
        # Optional OOD evaluation
        # -------------------------
        if ood_loader is not None:
            ood_metrics = evaluate_detr(
                model,
                ood_loader,
                device,
                w_emb=w_emb,
                w_conf=w_conf,
                use_cosine=use_cosine,
            )

            history["ood_loss"].append(ood_metrics["loss"])
            history["ood_emb"].append(ood_metrics["emb"])
            history["ood_conf"].append(ood_metrics["conf"])

            msg += (
                f" || OOD loss {ood_metrics['loss']:.4f}"
                f" | emb {ood_metrics['emb']:.4f}"
                f" | conf {ood_metrics['conf']:.4f}"
            )

        print(msg)

        if ckpt_path is not None:
            torch.save(
                {"model": model.state_dict(), "history": history},
                ckpt_path,
            )

        if metrics_path is not None:
            with open(metrics_path, "w") as f:
                json.dump(history, f, indent=2)

    return history