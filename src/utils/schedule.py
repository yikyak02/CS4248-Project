from transformers import get_linear_schedule_with_warmup
from torch.optim import Optimizer

def build_linear_warmup(optimizer: Optimizer, warmup_steps: int, total_steps: int):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )