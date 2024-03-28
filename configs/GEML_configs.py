from pydantic import Field

from configs.arguments import TrainingArguments


class GEMLConfig(TrainingArguments):
    p_interval: int = Field(1)
    embed_dim: int = Field(32)
    loss_p1: float = Field(0.5)
    loss_p2: float = Field(0.5)
