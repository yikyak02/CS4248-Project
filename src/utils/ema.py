# src/utils/ema.py
import torch

class EMA:
    """Minimal Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # Only track trainable params with requires_grad
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()

    def store(self, model):
        """Save current params to backup before applying EMA weights."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.backup[name] = param.detach().clone()

    @torch.no_grad()
    def copy_to(self, model):
        """Copy EMA weights to model (for eval/save)."""
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model):
        """Restore original (non-EMA) weights."""
        for name, param in model.named_parameters():
            if name in self.backup and param.requires_grad:
                param.data.copy_(self.backup[name].data)
        self.backup = {}
