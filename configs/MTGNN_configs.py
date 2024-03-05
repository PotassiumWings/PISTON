from pydantic import Field

from configs.arguments import TrainingArguments


class MTGNNConfig(TrainingArguments):
    layers: int = Field(2)
    residual_channels: int = Field(8)
    conv_channels: int = Field(8)
    skip_channels: int = Field(16)
    end_channels: int = Field(32)
    dilation_exponential: int = Field(2)
    gcn_depth: int = Field(2)
    propalpha: float = Field(0.05)
    layer_norm_affline: bool = Field(False)

    # dynamic graph
    use_graph_constructor: bool = Field(False)
    subgraph_size: int = Field(10)
    node_dim: int = Field(20)
    tanhalpha: float = Field(3)
    dropout: float = Field(0.2)
