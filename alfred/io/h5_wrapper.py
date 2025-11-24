"""

a extremly convenient way to loading h5 data as dict
"""

import h5py
import numpy as np


def save_h5(path, data):
    """Save a Python dict-like object into an HDF5 file."""
    with h5py.File(path, "w") as f:
        _save_to_group(f, data)


def load_h5(path):
    """Load an HDF5 file and return it as a Python dict."""
    with h5py.File(path, "r") as f:
        return _load_from_group(f)


# ---------------- internal helpers ---------------- #


def _save_to_group(h5grp, obj, name=None):
    if name is None:
        group = h5grp
    else:
        group = h5grp.require_group(name)

    if isinstance(obj, dict):
        group.attrs["__type__"] = "dict"
        for k, v in obj.items():
            _save_to_group(group, v, k)

    elif isinstance(obj, (list, tuple)):
        group.attrs["__type__"] = "list" if isinstance(obj, list) else "tuple"
        for i, v in enumerate(obj):
            _save_to_group(group, v, str(i))

    elif isinstance(obj, np.ndarray):
        group.create_dataset(name, data=obj)

    elif isinstance(obj, (int, float, str, np.number)):
        group.attrs["__scalar__"] = obj

    else:
        raise TypeError(f"Unsupported type: {type(obj)}")


def _load_from_group(h5grp):
    # scalar
    if "__scalar__" in h5grp.attrs:
        return _convert_scalar(h5grp.attrs["__scalar__"])

    # array
    if isinstance(h5grp, h5py.Dataset):
        return h5grp[()]

    # complex structure
    t = h5grp.attrs.get("__type__", None)

    if t == "dict":
        return {k: _load_from_group(h5grp[k]) for k in h5grp.keys()}

    elif t == "list":
        # keys are "0","1",...
        items = sorted(h5grp.keys(), key=lambda x: int(x))
        return [_load_from_group(h5grp[k]) for k in items]

    elif t == "tuple":
        items = sorted(h5grp.keys(), key=lambda x: int(x))
        return tuple(_load_from_group(h5grp[k]) for k in items)

    else:
        # if no type tag: treat as dataset
        if isinstance(h5grp, h5py.Dataset):
            return h5grp[()]
        return {k: _load_from_group(h5grp[k]) for k in h5grp.keys()}


def _convert_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x
