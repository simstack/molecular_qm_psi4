from typing import Dict

from odmantic import Field, Model

from simstack.models import simstack_model


@simstack_model
class NMRReferenceNuclei(Model):
    """1-based atom indexes in the reference molecule, keyed by element symbol.

    The isotropic GIAO shielding of that atom is σ_ref for the element.
    The chemical shift of a sample atom is σ_ref − σ.
    """

    field_name: str = "NMRReferenceNuclei"
    atom_index: Dict[str, int] = Field(
        ...,
        json_schema_extra={
            "description": (
                "Element symbol to 1-based atom index in the reference molecule. "
                "Example for TMS: H is one methyl hydrogen, C is one methyl carbon."
            )
        },
    )
