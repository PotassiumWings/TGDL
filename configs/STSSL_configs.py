from pydantic import Field

from configs.arguments import TrainingArguments


class STSSLConfig(TrainingArguments):
    d_model: int = Field(64)
    nmb_prototype: int = Field(6)
    shm_temp: float = Field(0.5)
    aug_percent: float = Field(0.1)
    temporal_percent: float = Field(0.5)
