from odmantic import Model, Field

from molecular_qm_psi4.models.ring_classifier import RingInfo
from simstack.models import simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

@simstack_model
class CreateLandscapeParams(Model):
    rng_seed: int = Field(default=42, description="Random number generator seed")
    max_iterations: int = Field(default=100, description="Maximum number of iterations")
    max_images: int = Field(default=10, description="Maximum number of images per iteration")
    ring_info: RingInfo = Field(default_factory=RingInfo)
    @classmethod
    def json_schema(cls):
        return cleaned_json_schema(cls)

    @classmethod
    def ui_schema(cls):
        return generate_ui_schema(cls)
