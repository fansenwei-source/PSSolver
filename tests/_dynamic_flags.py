"""Restore virtual backend attributes by value, never by deleting an attribute."""

from functools import partial


def set_dynamic_flag(request, target, name, value):
    """Register the actual getter value before setting a Torch-style flag.

    pytest 9.2 MonkeyPatch tracks the instance dictionary for non-descriptors.
    Torch's cuBLAS flags instead use __getattr__/__setattr__ and have no such
    dictionary entry. Their teardown must call the setter, not delattr.
    """
    previous = getattr(target, name)
    request.addfinalizer(partial(setattr, target, name, previous))
    setattr(target, name, value)
