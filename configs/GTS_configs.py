from pydantic import Field

from configs.arguments import TrainingArguments


class GTSConfig(TrainingArguments):
    max_diffusion_step: int = Field(2)
    cl_decay_steps: int = Field(1000)
    filter_type: str = Field("laplacian")
    num_rnn_layers: int = Field(1)
    rnn_units: int = Field(32)
    temperature: float = Field(0.5)
    epoch_use_regularization: int = Field(50)
    num_batches: int = Field(4600*32)
    kernal_size: int = Field(10)
    embedding_dim: int = Field(100)
