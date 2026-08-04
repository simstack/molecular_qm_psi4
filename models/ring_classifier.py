from typing import Optional, List

from odmantic import Model, Reference, Field, EmbeddedModel

from molecular_qm_models import MoleculeList
from molecular_qm_models.energy_units import MolecularEnergyUnit, MolecularEnergyUnitEnum
from simstack.models import simstack_model

@simstack_model
class RingInfo(EmbeddedModel):
    use_ring_id: bool = Field(default=False, description="Flag to decide to use a specific ring ID")
    ring_id: int = Field(default=0,
                              description="Unless specified otherwise, use the first ring")
    use_ring_indices: bool = Field(default=False, description="Flag to decide to manually specify the ring indices")
    ring_indices: List[int] = Field(default_factory=list, description="Atom indices of the ring (1-indexed, comma separated string or list)")
    list_rings: bool = Field(default=False, description="List detected rings and stop.")
    bond_scale: float = Field(default=1.25, description="Scale factor for automatic bond guessing.")

    @classmethod
    def json_schema(cls):
        from simstack.util.cleaned_json_schema import cleaned_json_schema

        schema = cleaned_json_schema(cls)
        schema["dependencies"] = {
            "use_ring_id": {
                "oneOf": [
                    {
                        "properties": {
                            "use_ring_id": {"enum": [False]}
                        }
                    },
                    {
                        "properties": {
                            "use_ring_id": {"enum": [True]},
                            "ring_id": {"type": "integer"}
                        },
                        "required": ["ring_id"]
                    }
                ]
            },
            "use_ring_indices": {
                "oneOf": [
                    {
                        "properties": {
                            "use_ring_indices": {"enum": [False]}
                        }
                    },
                    {
                        "properties": {
                            "use_ring_indices": {"enum": [True]},
                            "ring_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 1
                            }
                        },
                        "required": ["ring_indices"]
                    }
                ]
            }
        }
        return schema

    @classmethod
    def ui_schema(cls):
        from simstack.util.generate_ui_schema import generate_ui_schema

        return generate_ui_schema(cls)



@simstack_model
class RingConformerClassifier(Model):
    molecules: MoleculeList = Reference()
    rmsd_cutoff: float = Field(default=0.5, description="Mass-weighted RMSD cutoff in Angstrom.")
    energy_unit: MolecularEnergyUnit = Field(default=MolecularEnergyUnit(unit=MolecularEnergyUnitEnum.HARTREE), description="Energy unit in XYZ comment line.")
    use_energy_window: bool = Field(default=False, description="Whether to use an energy cutoff after classification.")
    energy_window: Optional[float] = Field(default=None, description="Optional energy cutoff in kcal/mol after classification.")
    ring_info: RingInfo = Field(default_factory=RingInfo)
    
    @classmethod
    def json_schema(cls):
        from simstack.util.cleaned_json_schema import cleaned_json_schema
        schema = cleaned_json_schema(cls)
        schema["dependencies"] = {
            "use_energy_window": {
                "oneOf": [
                    {
                        "properties": {
                            "use_energy_window": {"enum": [False]}
                        }
                    },
                    {
                        "properties": {
                            "use_energy_window": {"enum": [True]},
                            "energy_window": {"type": "number"}
                        },
                        "required": ["energy_window"]
                    }
                ]
            }
        }
        return schema

    @classmethod
    def ui_schema(cls):
        from simstack.util.generate_ui_schema import generate_ui_schema
        ui = generate_ui_schema(cls)
        # molecules is a reference, should be handled by generate_ui_schema
        # but if we want to be sure:
        return ui
