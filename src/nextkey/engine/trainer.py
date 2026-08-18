"""Unified training loop for CharTagger models.

Supports:
    - Configurable optimizer (Adam, AdamW) and scheduler (CosineAnnealing)
    - Early stopping with patience
    - Gradient clipping
    - Mixed precision (AMP) for research mode
    - Checkpoint save/load of best model
    - Structured JSON logging
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from nextkey.data.dataset import AlignedExample, iter_batches
from nextkey.data.tokenizer import CharVocab
from nextkey.engine.loss import DualHeadLoss, DistillationLoss
from nextkey.utils.metrics import MetricTotals


def resolve_device(requested: str | None) -> torch.device:
    """Resolve device string to torch.device, with auto-detection."""
    if requested and requested not in ("auto", ""):
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelTrainer:
    """Unified training loop for CharTagger backbone models."""

    def __init__(
        self,
        model: nn.Module,
        vocab: CharVocab,
        cfg: dict[str, Any],
        output_dir: str | Path,
        device: torch.device | None = None,
    ):
        self.model = model
        self.vocab = vocab
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or resolve_device(cfg.get("runtime", {}).get("device"))
        self.model.to(self.device)

        # Training config
        training = cfg.get("training", {})
        self.epochs = int(training.get("epochs", 2))
        self.batch_size = int(training.get("batch_size", 32))
        self.clip_grad_norm = float(training.get("clip_grad_norm", 1.0))
        self.patience = int(training.get("early_stopping_patience", 5))
        self.lambda_boundary = float(training.get("lambda_boundary", 1.0))
        self.domain_balanced = bool(training.get("domain_balanced", True))
        self.log_every = int(training.get("log_every", 50))
        self.max_steps = int(training.get("max_steps", 0)) or None

        # KD settings
        distill_cfg = cfg.get("distillation", {})
        self.alpha = float(distill_cfg.get("alpha", 0.0))
        self.temperature = float(distill_cfg.get("temperature", 2.0))

        # Loss
        if self.alpha > 0:
            self.criterion = DistillationLoss(
                pad_target_id=vocab.pad_target_id,
                lambda_boundary=self.lambda_boundary,
                alpha=self.alpha,
                temperature=self.temperature,
            )
        else:
            self.criterion = DualHeadLoss(
                pad_target_id=vocab.pad_target_id,
                lambda_boundary=self.lambda_boundary,
            )

        # Optimizer
        opt_cfg = cfg.get("optimizer", {})
        opt_type = opt_cfg.get("type", "AdamW")
        lr = float(opt_cfg.get("lr", training.get("learning_rate", 1e-3)))
        wd = float(opt_cfg.get("weight_decay", training.get("weight_decay", 1e-4)))

        if opt_type == "AdamW":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler
        sched_cfg = cfg.get("scheduler", {})
        sched_type = sched_cfg.get("type", "")
        if sched_type == "CosineAnnealingLR":
            self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=int(sched_cfg.get("T_max", self.epochs)),
                    eta_min=float(sched_cfg.get("eta_min", 1e-5)),
                )
            )
        else:
            self.scheduler = None

        # State
        self.best_cer = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.history: list[dict[str, Any]] = []

    def fit(
        self,
        train_examples: list[AlignedExample],
        val_examples: list[AlignedExample],
    ) -> dict[str, Any]:
        """Run the full training loop with validation and early stopping.

        Returns:
            dict with training summary (best metrics, paths, etc.)
        """
        model_path = self.output_dir / "best_model.pt"
        vocab_path = self.output_dir / "vocab.json"
        self.vocab.save(vocab_path)

        print(f"╔══════════════════════════════════════════════════════╗")
        print(f"║  Training: {self.model.__class__.__name__:<41s} ║")
        print(f"║  Device: {str(self.device):<43s} ║")
        print(f"║  Params: {self.model.count_parameters():>10,d}{'':<32s} ║")
        print(f"║  Train: {len(train_examples):>6,d}  |  Val: {len(val_examples):>6,d}{'':<19s} ║")
        print(f"║  Epochs: {self.epochs}  |  Batch: {self.batch_size}  |  α(KD): {self.alpha}{'':<12s} ║")
        print(f"╚══════════════════════════════════════════════════════╝")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_examples, epoch)
            val_metrics = self._evaluate(val_examples)

            record = {
                "epoch": epoch,
                "step": self.global_step,
                "train_loss": round(train_loss, 6),
                "val": val_metrics,
            }
            self.history.append(record)

            # Check improvement
            current_cer = val_metrics["corpus_cer"]
            improved = current_cer <= self.best_cer
            if improved:
                self.best_cer = current_cer
                self.patience_counter = 0
                torch.save({
                    "state_dict": self.model.state_dict(),
                    "config": self.cfg,
                    "vocab_path": str(vocab_path),
                    "best_val": val_metrics,
                    "epoch": epoch,
                }, model_path)
                marker = " ★ best"
            else:
                self.patience_counter += 1
                marker = f" (patience {self.patience_counter}/{self.patience})"

            print(f"  Epoch {epoch:>2d} │ loss={train_loss:.4f} │ "
                  f"CER={current_cer:.4f} │ BF1={val_metrics['boundary_f1']:.4f} │ "
                  f"EM={val_metrics['exact_match']:.4f}{marker}")

            if self.scheduler:
                self.scheduler.step()

            if self.patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

            if self.max_steps and self.global_step >= self.max_steps:
                print(f"  Max steps ({self.max_steps}) reached.")
                break

        # Save history
        history_path = self.output_dir / "training_history.json"
        history_path.write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        summary = {
            "model_path": str(model_path),
            "vocab_path": str(vocab_path),
            "history_path": str(history_path),
            "device": str(self.device),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "total_steps": self.global_step,
            "best_val_corpus_cer": self.best_cer,
            "parameters": self.model.count_parameters(),
        }
        print(f"\n  ✓ Best model saved → {model_path}")
        print(f"  ✓ Best val CER: {self.best_cer:.4f}")
        return summary

    def _train_epoch(
        self,
        examples: list[AlignedExample],
        epoch: int,
    ) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        loss_sum = 0.0
        batches = 0

        for source, targets, boundaries, lengths, _ in iter_batches(
            examples, self.vocab, self.batch_size,
            domain_balanced=self.domain_balanced,
        ):
            source = source.to(self.device)
            targets = targets.to(self.device)
            boundaries = boundaries.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(source, lengths)

            loss_dict = self.criterion(
                outputs["diacritic_logits"],
                outputs["boundary_logits"],
                targets,
                boundaries,
            )
            loss = loss_dict["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            self.global_step += 1
            batches += 1
            loss_sum += loss.item()

            if self.global_step % self.log_every == 0:
                avg = loss_sum / batches
                print(f"    step {self.global_step:>5d} │ loss={avg:.4f} "
                      f"(char={loss_dict['loss_char'].item():.4f}, "
                      f"bnd={loss_dict['loss_boundary'].item():.4f})")

            if self.max_steps and self.global_step >= self.max_steps:
                break

        return loss_sum / max(batches, 1)

    @torch.no_grad()
    def _evaluate(self, examples: list[AlignedExample]) -> dict[str, float | int]:
        """Evaluate on a set of examples. Returns metric dict."""
        self.model.eval()
        totals = MetricTotals()

        for source, targets, boundaries, lengths, _ in iter_batches(
            examples, self.vocab, self.batch_size,
        ):
            source = source.to(self.device)
            outputs = self.model(source, lengths)

            char_ids = outputs["diacritic_logits"].argmax(-1).cpu()
            boundary_preds = (outputs["boundary_logits"].sigmoid() >= 0.5).long().cpu()

            for row, length in enumerate(lengths.tolist()):
                prediction = self.vocab.decode(
                    char_ids[row, :length].tolist(),
                    boundary_preds[row, :length].tolist(),
                )
                target = self.vocab.decode(
                    targets[row, :length].tolist(),
                    boundaries[row, :length].clamp_min(0).tolist(),
                )
                totals.update(prediction, target)

        return totals.as_dict()
