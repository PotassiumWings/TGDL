from pydantic import Field

from configs.arguments import TrainingArguments


class MSDRConfig(TrainingArguments):
    max_diffusion_step: int = Field(2)
    filter_type: str = Field("dual_random_walk")
    num_rnn_layers: int = Field(2)
    rnn_units: int = Field(8)
    pre_k: int = Field(3)
    pre_v: int = Field(1)
