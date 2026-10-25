"""Small, portable requests and references for container-only density work."""
from odmantic import Model, Field
from simstack.models import simstack_model


@simstack_model
class PySCFArtifactLocation(Model):
    root: str
    path: str
    container_root: str | None = None


@simstack_model
class PySCFArtifact(Model):
    root: str
    path: str
    sha256: str
    kind: str
    schema_version: int = 1


@simstack_model
class PySCFDensityResponseInput(Model):
    source_root: str
    source_path: str
    container_source_root: str | None = None
    source_sha256: str
    group: str
    tolerance: float = Field(default=1e-6, gt=0)
