# -*- coding: utf-8 -*-
"""
learning/ewc.py
================
Elastic Weight Consolidation (EWC) — ngăn catastrophic forgetting khi fine-tune.

Original paper: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)

Ý tưởng:
  - Sau khi train xong task cũ (hoặc epoch trước), tính Fisher Information Matrix (FIM)
    cho từng parameter → đo lường "tầm quan trọng" của param đó đối với task cũ
  - Khi train task mới, thêm penalty: λ/2 * F * (θ - θ_old)²
    → parameter quan trọng (F cao) bị phạt mạnh nếu thay đổi quá nhiều

Integration:
  - Sau mỗi epoch/training run, gọi ewc.update_fisher(model, dataloader)
  - Khi train tiếp, thêm ewc.penalty(model) vào loss
  - Có thể dùng mode "online" (cập nhật liên tục) hoặc "offline" (tính 1 lần)

Memory: lưu FIM dưới dạng sparse (chỉ lưu diagonal) hoặc low-rank approximation.
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

log = logging.getLogger("learning.ewc")

# ─── EWC Config ────────────────────────────────────────────────────────────

DEFAULT_EWC_LAMBDA = 1000.0      # hệ số penalty — càng lớn = giữ knowledge cũ chặt hơn
DEFAULT_EWC_MODE   = "online"    # "online" | "offline"
DEFAULT_FISHER_SAMPLES = 500     # số sample dùng ước lượng Fisher (online mode)
DEFAULT_FISHER_BATCH_SIZE = 16   # batch size khi tính Fisher

# ─── EWC State ──────────────────────────────────────────────────────────────

@dataclass
class EWCState:
    """Lưu trữ EWC state để persistence."""
    param_names: List[str]
    param_means: List[List[float]]      # θ_old — giá trị param sau task cũ
    fisher_diag: List[List[float]]      # Diagonal Fisher Information
    param_shapes: List[List[int]]
    lambda_: float
    mode: str
    updated_at: str
    total_samples: int

    def to_dict(self) -> dict:
        return {
            "param_names": self.param_names,
            "param_means": self.param_means,
            "fisher_diag": self.fisher_diag,
            "param_shapes": self.param_shapes,
            "lambda": self.lambda_,
            "mode": self.mode,
            "updated_at": self.updated_at,
            "total_samples": self.total_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EWCState":
        return cls(**data)


# ─── EWC Module ────────────────────────────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation cho continual learning.
    
    Usage:
        ewc = EWC(lambda_=1000.0, mode="online")
        
        # Sau khi train task cũ xong:
        ewc.update_fisher(model, dataloader, num_samples=1000)
        ewc.save("ewc_state.json")
        
        # Khi train task mới:
        loss = task_loss + ewc.penalty(model)
        loss.backward()
    """

    def __init__(
        self,
        lambda_: float = DEFAULT_EWC_LAMBDA,
        mode: str = DEFAULT_EWC_MODE,
        fisher_samples: int = DEFAULT_FISHER_SAMPLES,
        fisher_batch_size: int = DEFAULT_FISHER_BATCH_SIZE,
        device: Optional[str] = None,
        state_path: Optional[Path] = None,
    ):
        self.lambda_ = lambda_
        self.mode = mode
        self.fisher_samples = fisher_samples
        self.fisher_batch_size = fisher_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # State
        self._param_names: List[str] = []
        self._param_means: Dict[str, torch.Tensor] = {}    # θ_old
        self._fisher_diag: Dict[str, torch.Tensor] = {}    # F (diagonal)
        self._param_shapes: Dict[str, List[int]] = {}
        self._total_samples = 0
        self._lock = threading.Lock()
        
        # Load state if exists
        if state_path and state_path.exists():
            self.load(state_path)

    # ─── Core: Fisher Information Estimation ──────────────────────────────────

    def update_fisher(
        self,
        model: nn.Module,
        dataloader,
        num_samples: Optional[int] = None,
        criterion: Optional[nn.Module] = None,
    ):
        """
        Cập nhật Fisher Information Matrix (diagonal approximation).
        
        Online mode:  F_new = (N_old * F_old + N_new * F_new) / (N_old + N_new)
        Offline mode: tính lại từ đầu.
        
        Args:
            model: model đã train xong task cũ
            dataloader: dữ liệu task cũ (để ước lượng Fisher)
            num_samples: số sample dùng ước lượng (None = self.fisher_samples)
            criterion: loss function (None = CrossEntropy)
        """
        num_samples = num_samples or self.fisher_samples
        criterion = criterion or nn.CrossEntropyLoss(reduction="sum")
        
        model.eval()
        model.to(self.device)
        
        # Accumulate gradients
        grad_accum = {name: torch.zeros_like(param, device=self.device)
                      for name, param in model.named_parameters()
                      if param.requires_grad}
        
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward
            outputs = model(**batch)
            loss = criterion(outputs, batch.get("labels", batch.get("input_ids")))
            
            # Backward
            model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher ≈ E[grad²])
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_accum[name] += param.grad.detach() ** 2
            
            samples_processed += batch.get("labels", batch.get("input_ids")).size(0)
        
        # Normalize: Fisher ≈ E[grad²] = sum(grad²) / N
        fisher_new = {name: g / max(samples_processed, 1) for name, g in grad_accum.items()}
        
        # Merge with existing Fisher
        with self._lock:
            if self.mode == "online" and self._fisher_diag:
                # Online: weighted average
                total_old = self._total_samples
                total_new = samples_processed
                total = total_old + total_new
                for name in fisher_new:
                    if name in self._fisher_diag:
                        self._fisher_diag[name] = (
                            self._fisher_diag[name] * total_old + fisher_new[name] * total_new
                        ) / total
                    else:
                        self._fisher_diag[name] = fisher_new[name]
            else:
                # Offline: replace
                self._fisher_diag = fisher_new
            
            # Update means (θ_old)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name not in self._param_means:
                        self._param_means[name] = param.detach().clone()
                    elif self.mode == "online":
                        # Online: weighted average of means
                        total_old = self._total_samples
                        total_new = samples_processed
                        total = total_old + total_new
                        self._param_means[name] = (
                            self._param_means[name] * total_old + param.detach() * total_new
                        ) / total
                    else:
                        self._param_means[name] = param.detach().clone()
                    self._param_shapes[name] = list(param.shape)
            
            self._total_samples += samples_processed
            self._param_names = list(self._param_means.keys())
            
            log.info(f"[EWC] Updated Fisher: {len(self._param_means)} params, "
                     f"{samples_processed} samples, total={self._total_samples}")

    def _move_batch_to_device(self, batch):
        """Move batch tensors to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return tuple(self._move_batch_to_device(x) for x in batch)
        return batch

    # ─── Penalty Computation ──────────────────────────────────────────────────

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Tính EWC penalty: λ/2 * Σ F_i * (θ_i - θ_old_i)²
        
        Returns: scalar tensor (để add vào loss)
        """
        if not self._fisher_diag:
            return torch.tensor(0.0, device=self.device)
        
        penalty = 0.0
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self._fisher_diag:
                continue
            fisher = self._fisher_diag[name]
            old_mean = self._param_means[name]
            diff = param - old_mean
            penalty += (fisher * diff ** 2).sum()
        
        return 0.5 * self.lambda_ * penalty

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path):
        """Lưu EWC state ra file JSON."""
        state = EWCState(
            param_names=self._param_names,
            param_means=[v.cpu().tolist() for v in self._param_means.values()],
            fisher_diag=[v.cpu().tolist() for v in self._fisher_diag.values()],
            param_shapes=[self._param_shapes[n] for n in self._param_names],
            lambda_=self.lambda_,
            mode=self.mode,
            updated_at=__import__("datetime").datetime.now().isoformat(),
            total_samples=self._total_samples,
        )
        path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
        log.info(f"[EWC] Saved state to {path} ({len(self._param_names)} params)")

    def load(self, path: Path):
        """Load EWC state từ file JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        state = EWCState.from_dict(data)
        
        self._param_names = state.param_names
        self._param_means = {
            name: torch.tensor(mean, device=self.device)
            for name, mean in zip(state.param_names, state.param_means)
        }
        self._fisher_diag = {
            name: torch.tensor(fisher, device=self.device)
            for name, fisher in zip(state.param_names, state.fisher_diag)
        }
        self._param_shapes = dict(zip(state.param_names, state.param_shapes))
        self.lambda_ = state.lambda_
        self.mode = state.mode
        self._total_samples = state.total_samples
        
        log.info(f"[EWC] Loaded state from {path} ({len(self._param_names)} params, "
                 f"{state.total_samples} samples)")

    # ─── Utilities ────────────────────────────────────────────────────────────

    def get_importance(self, name: str) -> Optional[float]:
        """Trả về tầm quan trọng (Fisher) của 1 parameter."""
        if name in self._fisher_diag:
            return float(self._fisher_diag[name].mean().item())
        return None

    def get_top_important(self, top_k: int = 10) -> List[tuple]:
        """Top-K parameter quan trọng nhất."""
        importances = [(name, self.get_importance(name)) for name in self._param_names]
        importances.sort(key=lambda x: x[1] or 0, reverse=True)
        return importances[:top_k]

    def reset(self):
        """Reset EWC state."""
        with self._lock:
            self._param_names = []
            self._param_means = {}
            self._fisher_diag = {}
            self._param_shapes = {}
            self._total_samples = 0
            log.info("[EWC] State reset")


# ─── Trainer Integration Helper ──────────────────────────────────────────────

def create_ewc_callback(
    ewc: "EWC",
    dataloader,
    num_samples: int = 500,
    every_n_epochs: int = 1,
) -> callable:
    """
    Tạo callback để gọi sau mỗi epoch trong training.
    
    Usage:
        ewc = EWC(lambda_=1000)
        callback = create_ewc_callback(ewc, train_dataloader, num_samples=500)
        
        trainer = Trainer(
            ...,
            callbacks=[callback],
        )
    """
    def callback(trainer, model, epoch, *args, **kwargs):
        if epoch % every_n_epochs == 0:
            log.info(f"[EWC Callback] Updating Fisher at epoch {epoch}")
            ewc.update_fisher(model, dataloader, num_samples=num_samples)
            ewc.save(Path("learning/ewc_state.json"))
    
    return callback


def add_ewc_to_trainer(trainer, ewc: "EWC"):
    """
    Monkey-patch trainer.compute_loss để thêm EWC penalty.
    
    Usage:
        ewc = EWC(lambda_=1000)
        add_ewc_to_trainer(trainer, ewc)
        trainer.train()
    """
    original_compute_loss = trainer.compute_loss
    
    def compute_loss_with_ewc(model, inputs, return_outputs=False):
        loss, outputs = original_compute_loss(model, inputs, return_outputs)
        ewc_penalty = ewc.penalty(model)
        total_loss = loss + ewc_penalty
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    trainer.compute_loss = compute_loss_with_ewc
    log.info("[EWC] Monkey-patched trainer.compute_loss with EWC penalty")


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_ewc_from_config(config: dict, device: Optional[str] = None) -> "EWC":
    """Tạo EWC từ config dict."""
    return EWC(
        lambda_=config.get("lambda", 1000.0),
        mode=config.get("mode", "online"),
        fisher_samples=config.get("fisher_samples", 500),
        fisher_batch_size=config.get("fisher_batch_size", 16),
        device=device,
    )


if __name__ == "__main__":
    # Quick test
    import torch.nn as nn
    
    model = nn.Linear(10, 2)
    ewc = EWC(lambda_=1000)
    
    # Fake data
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)
    
    criterion = nn.CrossEntropyLoss()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    ewc.update_fisher(model, loader, num_samples=32)
    print(f"Penalty: {ewc.penalty(model).item()}")
    ewc.save(Path("test_ewc.json"))