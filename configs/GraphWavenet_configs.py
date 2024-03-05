from pydantic import Field

from configs.arguments import TrainingArguments


class GraphWavenetConfig(TrainingArguments):
    blocks: int = Field(3)
    layers: int = Field(2)
    hidden_size: int = Field(8)
    kernel_size: int = Field(2)
    dropout: float = Field(0.2)
