from pydantic import BaseModel, Field


class TrainingArguments(BaseModel):
    dataset_name: str = Field("BikeNYC")
    # data feature
    input_len: int = Field(19)  # Time step len
    output_len: int = Field(1)  # next step prediction, not changeable
    num_nodes: int = Field(128)
    c_in: int = Field(2)  # feature dim
    c_out: int = Field(2)

    batch_size: int = Field(32)

    # for trainer
    show_period: int = Field(160)  # batches to evaluate
    accumulate_period: int = Field(32)  # batches to step loss

    learning_rate: float = Field(1e-3)
    num_epoches: int = Field(100)
    early_stop_batch: int = Field(100000)
    optimizer: str = Field("Adam")

    scaler: str = Field("Standard")

    # for GPSTP
    num_subgraphs: int = Field(2)
    softmax_graphs: bool = Field(False)

    gumbel_softmax: bool = Field(False)
    start_temperature: float = Field(1.0)
    decay_temperature: float = Field(0.99)

    num_heads: int = Field(4)
    num_supports: int = Field(1)

    seed: int = Field(0)
    use_dwa: bool = Field(False)
    sparse: bool = Field(False)
    compare: bool = Field(False)  # only use G as input, no decomposition

    # for dtw distance
    sigma: float = Field(0.1)
    thres: float = Field(0.6)

    loss_1: float = Field(1.0)
    loss_2: float = Field(1.0)
    loss_3: float = Field(1000.0)
    loss_4: float = Field(100.0)
    loss_5: float = Field(1.0)
    loss: str = Field("MAE")
    use_origin_x: bool = Field(False)
    loss_delta: float = Field(5.0)
    require_sum: bool = Field(False)  # loss4 requires edges in all graphs equals the whole graph(1)
    require_le: bool = Field(False)  # loss4 only requires sum of subgraphs <= origin graph 这个没用，一直是 false
    remove_backward: bool = Field(False)
    straight: bool = Field(False)

    lamb: float = Field(0.46)  # inflow loss percent
    ignore_graph_epoch: int = Field(100)  # after which only optimize loss_1 and loss_2
    nndT: int = Field(10)
    clip: int = Field(5)

    dropout: float = Field(0.2)
    nnd: bool = Field(False)
    orth_reg: bool = Field(True)  # 正交正则化
    traditional: bool = Field(False)  # 传统方法分解
    spectral_clustering: bool = Field(False)  # 谱聚类
    start_from_pre: bool = Field(False)

    load: str = Field("")

    mae_mask: int = Field(5)
    debug: bool = Field(False)
    vertex: int = Field(150)
    visualize: bool = Field(False)  # False 的时候只获得实验结果
    tsne: int = Field(-1)

    continue_training_epoch: int = Field(-1)
    fast_eval: int = Field(-1)
    subgraph_grad: bool = Field(True)

    st_encoder: str = Field("STGCN")
    save_graph: bool = Field(False)
