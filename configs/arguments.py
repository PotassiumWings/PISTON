from pydantic import BaseModel, Field


class TrainingArguments(BaseModel):
    dataset_name: str = Field("NYC-TOD5")
    # data feature
    input_len: int = Field(48)  # Time step len
    output_len: int = Field(12)  # next step prediction, not changeable

    num_nodes: int = Field(75)

    c_in: int = Field(1)  # feature dim
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
    rsvd: bool = Field(True)
    random_svd_k: int = Field(15)
    use_rsvd_emb: bool = Field(False)  # use u, sig, v as emb
    p: int = Field(2)
    q: int = Field(5)

    # mask percent
    mask: bool = Field(False)
    mask_percent: float = Field(0.3)
    recover: bool = Field(False)
    recover_lamb: float = Field(1)

    # do contrastive learning
    contra: bool = Field(False)
    contra_lamb: float = Field(1)

    use_dwa: bool = Field(False)
    loss_lamb: float = Field(1)

    disable_ssl_epoch: int = Field(100)

    # correlation encoder
    adp_emb: int = Field(10)  # adaptive graph embedding length
    layers: int = Field(4)
    d_model: int = Field(16)
    d_ff: int = Field(16)
    n_head: int = Field(2)
    d_encoder: int = Field(16)
    order: int = Field(2)  # graph convolution cheby order

    seed: int = Field(0)
    load: str = Field("")
    dropout: float = Field(0.2)

    mae_mask: int = Field(0)

    # for traffic forecasting
    tradition_problem: bool = Field(False)
    origin_c_in: int = Field(2)
