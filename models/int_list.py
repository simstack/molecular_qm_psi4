from typing import Iterator, List

from odmantic import Field, Model

from simstack.models.simstack_model import simstack_model
from simstack.util.generic_list_mixin import GenericListMixin


@simstack_model
class IntList(Model, GenericListMixin[int]):
    field_name: str = "IntList"
    elements: List[int] = Field(default_factory=list, description="List of integers")

    def __iter__(self) -> Iterator[int]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)
