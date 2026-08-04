from typing import Optional, List
from odmantic import Model, Field, Reference
from simstack.models import simstack_model, DataSet
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

@simstack_model
class DihedralPuckeringFilterInput(Model):
    dataset: DataSet = Reference()
    ring_indices: Optional[List[int]] = Field(default=None, description="Atom indices of the ring (1-indexed)")
    rmsd_cutoff: float = Field(default=0.5, description="RMSD cutoff for diversity filtering")
    theta_cutoff: float = Field(default=12.0, description="Theta cutoff for diversity filtering")
    phi_cutoff: float = Field(default=20.0, description="Phi cutoff for diversity filtering")
    dihedral_cutoff: float = Field(default=25.0, description="Dihedral cutoff for diversity filtering")
    max_pairs_per_chair: int = Field(default=3, description="Maximum number of twist-boat pairs per chair")
    max_pair_score: Optional[float] = Field(default=None, description="Maximum allowed pair score")

    @classmethod
    def json_schema(cls):
        return cleaned_json_schema(cls)

    @classmethod
    def ui_schema(cls):
        return generate_ui_schema(cls)
