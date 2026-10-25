"""HDF5 wire format for container-only scientific operations (no pickle)."""
import hashlib
import json
from pathlib import Path
import uuid

import h5py
import numpy as np


def contained(root, relative):
    root = Path(root).expanduser()
    if not root.is_absolute():
        raise ValueError('artifact root must be an absolute native path')
    root = root.resolve()
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('artifact path must be root-relative without traversal')
    path = (root/path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('artifact path escapes its root')
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_artifact(location, kind, arrays, metadata):
    from molecular_qm_psi4.models.density_artifacts import PySCFArtifact
    path = contained(location.container_root or location.root, location.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name('.'+uuid.uuid4().hex+'.tmp')
    try:
        with h5py.File(temporary, 'x') as artifact:
            artifact.attrs.update(schema_version=1, kind=kind)
            artifact.create_dataset('metadata_json', data=json.dumps(metadata, allow_nan=False),
                                    dtype=h5py.string_dtype('utf-8'))
            for name, value in arrays.items():
                array = np.asarray(value)
                options = {'compression':'gzip', 'shuffle':True} if array.size and array.nbytes >= 4096 else {}
                artifact.create_dataset(name, data=array, **options)
        # A unique requested output location belongs to one child task.
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return PySCFArtifact(root=location.root, path=location.path,
                         sha256=sha256(path), kind=kind)


def basis_metadata(mol):
    if mol.cart:
        raise ValueError('Cartesian density layouts are unsupported')
    offsets = mol.ao_loc_nr()
    atoms = []
    for atom, (s0,s1,_,_) in enumerate(mol.aoslice_by_atom()):
        shells = []
        for shell in range(s0,s1):
            l = int(mol.bas_angular(shell))
            if l > 2:
                raise ValueError('only spherical s/p/d density layouts are supported')
            shells.append(dict(l=l, contractions=int(mol.bas_nctr(shell)), start=int(offsets[shell])))
        atoms.append(dict(atomic_number=int(mol.atom_charge(atom)), shells=shells))
    return dict(atoms=atoms, nao=int(mol.nao))


def fitted_coefficients(mol, dm):
    dm = np.asarray(dm)
    if dm.ndim == 3:
        dm = dm.sum(0)
    return np.linalg.solve(mol.intor('int1e_ovlp'), np.einsum(
        'ij,ijk->k', dm, mol.intor('int3c1e', shls_slice=(0,mol.nbas)*3)))


def native_angular_samples():
    from pyscf import gto
    points = np.random.default_rng(7351).normal(size=(64,3))
    points /= np.linalg.norm(points,axis=1,keepdims=True)
    result = {'angular_points':points}
    for l in range(3):
        mol = gto.M(atom='H 0 0 0', spin=1, basis={'H':[[l,[1.,1.]]]}, verbose=0)
        result[f'angular_l{l}'] = mol.eval_gto('GTOval_sph',points)
    return result


def response_artifact(request, location, memory_mb, threads):
    import pyscf
    from pyscf import gto, lib
    from molecular_qm_psi4.util.density_calculation import reload_wavefunction
    from molecular_qm_psi4.util.density_response import descriptor_response
    source = contained(request.container_source_root or request.source_root, request.source_path)
    if sha256(source) != request.source_sha256:
        raise ValueError('source shard checksum mismatch')
    old_threads = lib.num_threads()
    try:
        lib.num_threads(threads)
        with h5py.File(source,'r') as shard:
            group = shard[request.group]
            mf = reload_wavefunction(group['reference'])
            mf.max_memory = memory_mb
            mol = mf.mol
            target = gto.loads(group['target/wavefunction/molecule_json'].asstr()[()])
            if (mol._basis != target._basis or mol.cart != target.cart
                    or not np.array_equal(mol.atom_charges(),target.atom_charges())
                    or not np.allclose(mol.atom_coords(),target.atom_coords(),atol=1e-12)):
                raise ValueError('reference and target basis/geometry differ')
            if group['target'].attrs['method'].lower().replace(' ','') not in ('pbe0','pbe1pbe'):
                raise ValueError('expected a PBE0 target')
            values, jacobian = descriptor_response(mf,tolerance=request.tolerance)
            target_values = fitted_coefficients(mol,group['target/density_matrix'][...])
            arrays = dict(reference_coefficients=values, target_coefficients=target_values,
                          descriptor_jacobian=jacobian, positions_bohr=mol.atom_coords(),
                          atomic_numbers=mol.atom_charges(), **native_angular_samples())
            metadata = dict(basis=basis_metadata(mol), pyscf_version=pyscf.__version__,
                            source_sha256=request.source_sha256, group=request.group,
                            tolerance=request.tolerance)
        return write_artifact(location,'density_response',arrays,metadata)
    finally:
        lib.num_threads(old_threads)
