"""One explicit evaluation owns a short-lived table of physical derivatives."""


class StepDerivativeTable(dict):
    """Reuse identical (field, axis orders, basis signature) requests."""

    def resolve(self, key, compute):
        if key not in self:
            self[key] = compute()
        return self[key]
