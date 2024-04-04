from pydantic import Field

from configs.arguments import TrainingArguments


class AGCRNConfig(TrainingArguments):
    rnn_units: int = Field(64)
    embed_dim: int = Field(10)
    num_layers: int = Field(2)
    cheb_order: int = Field(2)
