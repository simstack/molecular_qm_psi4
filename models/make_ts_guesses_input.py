from typing import Optional, List
from odmantic import Model, Field, Reference
from simstack.models import simstack_model, DataSet
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

@simstack_model
class MakeTSGuessesInput(Model):
    dataset: DataSet = Reference()
    ring_indices: Optional[List[int]] = Field(default=None, description="Atom indices of the ring (1-indexed, comma separated string or list)")
    nimages: int = Field(default=101, description="Number of images for interpolation")
    nkeep: int = Field(default=5, description="Number of TS candidates to keep per pair")

    @classmethod
    def json_schema(cls):
        return cleaned_json_schema(cls)

    @classmethod
    def ui_schema(cls):
        return generate_ui_schema(cls)
