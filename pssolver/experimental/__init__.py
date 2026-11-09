"""Explicit opt-in pilots and independent references.

Retired shadow/canary implementations and frozen stage tools are not package
exports. Independent reference modules and boundary pilots retain their defining
module paths; importing this namespace has no eager runtime side effects.
"""

__all__: list[str] = []
