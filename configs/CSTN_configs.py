from pydantic import Field

from configs.arguments import TrainingArguments


class CSTNConfig(TrainingArguments):
    n_layers: int = Field(3)
    c_lt: int = Field(75)
