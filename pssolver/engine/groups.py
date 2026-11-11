"""Packed transform selection and publication, preserving v1 gather/scatter."""


def _get_field_index(name, name_to_idx):
    """Return the index for a field name, raising error if not present."""
    if name not in name_to_idx:
        raise KeyError(f"Field name '{name}' not found.")
    return name_to_idx[name]


def _normalize_field_key(key, name_to_idx):
    if isinstance(key, str):
        return _get_field_index(key, name_to_idx)
    if isinstance(key, int):
        return key
    raise KeyError("Field key must be a field name (str) or index (int).")


def _normalize_field_indices(keys, name_to_idx, component_count):
    try:
        indices = [_normalize_field_key(key, name_to_idx) for key in keys]
    except TypeError as exc:
        raise TypeError("Field indices must be an iterable of names or indices.") from exc
    if len(indices) == 0:
        raise ValueError("At least one field index is required.")
    for index in indices:
        if index < 0 or index >= component_count:
            raise IndexError(f"Field index {index} is out of range.")
    return indices


def _transform_group_indexer(indices, indexing):
    if indexing == "contiguous_slice":
        start = indices[0]
        if indices == list(range(start, start + len(indices))):
            return slice(start, start + len(indices)), "contiguous_slice"
    return indices, "advanced"


def select_group(tensor, keys, name_to_idx, component_count, indexing):
    indices = _normalize_field_indices(keys, name_to_idx, component_count)
    indexer, _ = _transform_group_indexer(indices, indexing)
    return tensor[indexer]


def store_group(destination, keys, values, name_to_idx, component_count, indexing):
    indices = _normalize_field_indices(keys, name_to_idx, component_count)
    indexer, effective = _transform_group_indexer(indices, indexing)
    if effective == "contiguous_slice":
        destination[indexer].copy_(values)
    else:
        destination[indexer] = values
