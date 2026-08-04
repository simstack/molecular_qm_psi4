from enum import Enum
from typing import Optional, List
from odmantic import Model, Field, Reference, EmbeddedModel
from molecular_qm_models.molecule import Molecule, MoleculeList
from simstack.core.hash import hash_value
from simstack.models import simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

class CrestLevelOfTheoryEnum(str, Enum):
    GFN2_XTB = "gfn2"
    GFN1_XTB = "gfn1"
    GFN0_XTB = "gfn0"
    GFN_FF     = "gff"
    GFN2_GFNFF = "gfn2-gfnff"

class CrestMDAlgoEnum(str, Enum):
    METADYNAMICS = "mtd"
    MOLECULAR_DYNAMICS = "md"

class CrestSolventEnum(str, Enum):
    NONE = "none"
    GBSA = "gbsa"
    ALPB = "alpb"

class CrestSolvents(str, Enum):
    ACETONE = "acetone"
    ACETONITRILE = "acetonitrile"
    ANILINE = "aniline"
    BENZALDEHYDE = "benzaldehyde"
    BENZENE = "benzene"
    CH2CL2 = "ch2cl2"
    CHCL3 = "chcl3"
    CS2 = "cs2"
    DMF = "dmf"
    DMSO = "dmso"
    ETHER = "ether"
    ETHYLACETATE = "ethylacetate"
    H2O = "h2o"
    METHANOL = "methanol"
    NITROMETHANE = "nitromethane"
    N_OCTANOL = "n-octanol"
    PHENOL = "phenol"
    TOLUENE = "toluene"
    THF = "thf"

class CrestOptLevelEnum(str, Enum):
    CRUDE = "crude"
    SLOPPY = "sloppy"
    LOOSE = "loose"
    LAX = "lax"
    NORMAL = "normal"
    TIGHT = "tight"
    VTIGHT = "vtight"
    EXTREME = "extreme"

class CrestClusterArgEnum(str, Enum):
    LOOSE = "loose"
    NORMAL = "normal"
    TIGHT = "tight"
    VTIGHT = "vtight"
    INCR = "incr"
    TIGHTINCR = "tightincr"
    VTIGHTINCR = "vtightincr"

@simstack_model
class CrestLevelOfTheory(EmbeddedModel):
    method: CrestLevelOfTheoryEnum = Field(CrestLevelOfTheoryEnum.GFN2_XTB, description="Semi-empirical method to use")
    use_solvent: CrestSolventEnum = Field(CrestSolventEnum.NONE, description="Solvent model to use")

    solvent: Optional[CrestSolvents] = Field(None, description="Solvent name (e.g., benzene, h2o)")
    charge: int = Field(0, description="Net charge of the molecule")
    multiplicity: int = Field(1, description="Spin multiplicity (2S+1)")

    def complex_hash(self):
        return hash_value(f"{self.method}_{self.use_solvent}_{self.solvent}_{self.charge}_{self.multiplicity}")

    @classmethod
    def json_schema(cls):
        schema = cleaned_json_schema(cls)
        prop_schemas = schema.get("properties", {})
        
        # Pop solvent schema so it can be handled conditionally
        solvent_schema = prop_schemas.pop("solvent", None)

        schema["dependencies"] = {
            "use_solvent": {
                "oneOf": [
                    {
                        "properties": {
                            "use_solvent": {"const": CrestSolventEnum.NONE}
                        }
                    },
                    {
                        "properties": {
                            "use_solvent": {"const": CrestSolventEnum.GBSA},
                            "solvent": solvent_schema
                        },
                        "required": ["solvent"]
                    },
                    {
                        "properties": {
                            "use_solvent": {"const": CrestSolventEnum.ALPB},
                            "solvent": solvent_schema
                        },
                        "required": ["solvent"]
                    }
                ]
            }
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["solvent"] = {
            "ui:condition": {
                "use_solvent": [CrestSolventEnum.GBSA, CrestSolventEnum.ALPB]
            }
        }
        return ui

@simstack_model
class CrestMDOpt(EmbeddedModel):
    temp: float = Field(300.0, description="Temperature in K for the MD/MTD")
    time: float = Field(5.0, description="Length of the MD/MTD simulation in ps")
    timestep: float = Field(1.0, description="Time step for the MD/MTD simulation in fs")
    shake: int = Field(2, description="SHAKE algorithm level (0: none, 1: H-only, 2: all bonds)")

@simstack_model
class CrestConfSearchOpt(EmbeddedModel):
    opt_level: CrestOptLevelEnum = Field(CrestOptLevelEnum.NORMAL, description="Optimization level (affects convergence thresholds)")
    ewin: float = Field(6.0, description="Energy window in kcal/mol for conformer selection")
    rmsd: float = Field(0.125, description="RMSD threshold in Angstrom for conformer pruning")
    v4: bool = Field(False, description="Use the new iMTD-GC workflow (v4)")
    quick: bool = Field(False, description="Use a quicker search (fewer MTD runs)")

@simstack_model
class CrestEnsembleSortingOpt(EmbeddedModel):
    use_sorting: bool = Field(False, description="Use non standard sorting")
    ewin: float = Field(6.0, description="Energy threshold in kcal/mol")
    rthr: float = Field(0.125, description="RMSD threshold in Angstrom")
    ethr: float = Field(0.05, description="Energy threshold between conformer pairs in kcal/mol")
    bthr: float = Field(0.01, description="Lower bound for rotational constant threshold")
    pthr: float = Field(0.0, description="Boltzmann population threshold (currently unused)")
    nmr: bool = Field(False, description="Determine and print NMR-equivalencies")
    athr: float = Field(0.04, description="Similarity threshold for NMR-equivalent atoms")
    temp: float = Field(298.15, description="Temperature for Boltzmann populations in K")
    esort: bool = Field(False, description="Sort only based on energy")
    nowr: bool = Field(False, description="Skip writing new ensemble files")
    subrmsd: bool = Field(False, description="Compare only parts of the structure included in the MTD bias")

    @classmethod
    def json_schema(cls):
        schema = cleaned_json_schema(cls)
        prop_schemas = schema.get("properties", {})
        
        fields = ["ewin", "rthr", "ethr", "bthr", "pthr", "nmr", "athr", "temp", "esort", "nowr", "subrmsd"]
        sub_schemas = {f: prop_schemas.pop(f) for f in fields if f in prop_schemas}

        schema["dependencies"] = {
            "use_sorting": {
                "oneOf": [
                    {
                        "properties": {
                            "use_sorting": {"const": False}
                        }
                    },
                    {
                        "properties": {
                            "use_sorting": {"const": True},
                            **sub_schemas
                        }
                    }
                ]
            }
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["use_sorting"] = {
            "ui:widget": "checkbox",
            "ui:title": "use non standard sorting"
        }
        fields = ["ewin", "rthr", "ethr", "bthr", "pthr", "nmr", "athr", "temp", "esort", "nowr", "subrmsd"]
        for field in fields:
            ui[field] = {
                "ui:condition": {
                    "use_sorting": True
                }
            }
        return ui

@simstack_model
class CrestPCAClusteringOpt(EmbeddedModel):
    cluster: bool = Field(False, description="Use non standard clustering")
    cluster_num: Optional[int] = Field(None, description="Number of clusters to produce (leave empty for autonomous mode)")
    cluster_level: Optional[CrestClusterArgEnum] = Field(None, description="Clustering level for autonomous mode")
    pccap: int = Field(0, description="Limit number of principal components (0 = no limit)")
    nopcmin: bool = Field(False, description="Remove lower bound for principal component contribution")
    pcaex: Optional[str] = Field(None, description="Ignore atoms listed in principle component setup (atomlist format)")

    @classmethod
    def json_schema(cls):
        schema = cleaned_json_schema(cls)
        prop_schemas = schema.get("properties", {})
        
        # Pull out fields to make them conditional on 'cluster'
        cluster_num = prop_schemas.pop("cluster_num", None)
        if cluster_num and "anyOf" in cluster_num:
            for opt in cluster_num["anyOf"]:
                if opt.get("type") == "integer":
                    opt["title"] = "Cluster number"
                elif opt.get("type") == "null":
                    opt["title"] = "standard"

        cluster_level = prop_schemas.pop("cluster_level", None)
        if cluster_level and "anyOf" in cluster_level:
            for opt in cluster_level["anyOf"]:
                if opt.get("type") == "null":
                    opt["title"] = "Automatic"

        pccap = prop_schemas.pop("pccap", None)
        nopcmin = prop_schemas.pop("nopcmin", None)
        
        pcaex = prop_schemas.pop("pcaex", None)
        if pcaex and "anyOf" in pcaex:
            for opt in pcaex["anyOf"]:
                if opt.get("type") == "string":
                    opt["title"] = "Manual pcaex"
                elif opt.get("type") == "null":
                    opt["title"] = "Automatic pcaex"

        schema["dependencies"] = {
            "cluster": {
                "oneOf": [
                    {
                        "properties": {
                            "cluster": {"const": False}
                        }
                    },
                    {
                        "properties": {
                            "cluster": {"const": True},
                            "cluster_num": cluster_num,
                            "cluster_level": cluster_level,
                            "pccap": pccap,
                            "nopcmin": nopcmin,
                            "pcaex": pcaex
                        }
                    }
                ]
            }
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["cluster"] = {
            "ui:widget": "checkbox",
            "ui:title": "use non standard clustering"
        }
        for field in ["cluster_num", "cluster_level", "pccap", "nopcmin", "pcaex"]:
            ui[field] = {
                "ui:condition": {
                    "cluster": True
                }
            }
        return ui

@simstack_model
class CrestInput(Model):
    molecule: Molecule = Reference()
    charge: int = Field(0, description="Net charge of the molecule")
    multiplicity: int = Field(1, description="Spin multiplicity (2S+1)")
    
    level_of_theory: CrestLevelOfTheory = Field(default_factory=CrestLevelOfTheory)
    md_options: CrestMDOpt = Field(default_factory=CrestMDOpt)
    conf_search_options: CrestConfSearchOpt = Field(default_factory=CrestConfSearchOpt)
    ensemble_sorting_options: CrestEnsembleSortingOpt = Field(default_factory=CrestEnsembleSortingOpt)
    pca_clustering_options: CrestPCAClusteringOpt = Field(default_factory=CrestPCAClusteringOpt)
    
    additional_keywords: Optional[str] = Field(None, description="Any additional CREST command line arguments")

    @classmethod
    def json_schema(cls):
        schema = cleaned_json_schema(cls)
        # Rename "Option 1" (string) and "Option 2" (null) for additional_keywords
        if "additional_keywords" in schema["properties"]:
            kw_schema = schema["properties"]["additional_keywords"]
            if "anyOf" in kw_schema:
                for opt in kw_schema["anyOf"]:
                    if opt.get("type") == "string":
                        opt["title"] = "Additional Keywords"
                    elif opt.get("type") == "null":
                        opt["title"] = "No Additional Keywords"
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["ui:order"] = [
            "molecule",
            "charge",
            "multiplicity",
            "level_of_theory",
            "md_options",
            "conf_search_options",
            "ensemble_sorting_options",
            "pca_clustering_options",
            "additional_keywords"
        ]
        return ui_schema


@simstack_model
class XTBInput(Model):
    molecules: MoleculeList = Reference()
    level_of_theory: CrestLevelOfTheory = Field(default_factory=CrestLevelOfTheory)
    compute_gradients: bool = Field(False, description="Whether to compute gradients")
    additional_keywords: Optional[str] = Field(None, description="Any additional CREST command line arguments")

    @classmethod
    def json_schema(cls):
        schema = cleaned_json_schema(cls)
        # Rename "Option 1" (string) and "Option 2" (null) for additional_keywords
        if "additional_keywords" in schema["properties"]:
            kw_schema = schema["properties"]["additional_keywords"]
            if "anyOf" in kw_schema:
                for opt in kw_schema["anyOf"]:
                    if opt.get("type") == "string":
                        opt["title"] = "Additional Keywords"
                    elif opt.get("type") == "null":
                        opt["title"] = "No Additional Keywords"
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["ui:order"] = [
            "molecules",
            "level_of_theory",
            "compute_gradients",
            "additional_keywords"
        ]
        return ui_schema
