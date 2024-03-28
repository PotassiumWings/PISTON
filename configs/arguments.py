from pydantic import BaseModel, Field


class TrainingArguments(BaseModel):
    dataset_name: str = Field("NYC-TOD")
    # data feature
    input_len: int = Field(48)  # Time step len
    output_len: int = Field(1)  # next step prediction, not changeable
    num_nodes: int = Field(75)
    c_in: int = Field(1)  # feature dim
    c_hid: int = Field(8)  # hidden dim
    c_out: int = Field(1)  # out dim

    batch_size: int = Field(32)
    early_stop_batch: int = Field(10000)

    # for trainer
    show_period: int = Field(160)  # batches to evaluate
    accumulate_period: int = Field(32)  # batches to step loss

    learning_rate: float = Field(1e-3)
    num_epoches: int = Field(100)
    optimizer: str = Field("Adam")

    scaler: str = Field("Standard")

    # for decomposing
    p: int = Field(2)
    q: int = Field(1)
    node_emb: int = Field(10)

    seed: int = Field(0)
    load: str = Field("")

    mae_mask: int = Field(5)
    st_encoder: str = Field("STGCN")
    is_od_model: bool = Field(False)
