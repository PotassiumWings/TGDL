from pydantic import Field

from configs.arguments import TrainingArguments


class GraphWavenetConfig(TrainingArguments):
    blocks: int = Field(5)
    layers: int = Field(3)
    hidden_size: int = Field(8)
    kernel_size: int = Field(2)
