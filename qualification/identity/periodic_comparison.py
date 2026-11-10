"""Only S-25 section 9's two explicitly approved comparison exclusions."""

from copy import deepcopy

from qualification.identity.analyze import compact


def comparable_periodic(record):
    """Copy for comparison; never alter collected raw records or runtime state."""
    value = deepcopy(record)
    if value["scene"]["family"] != "periodic" or value["disposition"] != "accepted":
        return value
    for identity in (value.get("identity"), value.get("trace", {}).get("identity")):
        if identity is not None:
            identity["runtime"].pop("adapter")
    trace = value["trace"]
    runs = [trace["steps"], *trace["restarts"].values()]
    for rows in runs:
        for row in rows:
            for item in row.values():
                if item is not None and "progress" in item:
                    item["progress"].pop("constraints_current")
    for contract in trace["restart_contracts"].values():
        retained = []
        for item in contract["raw_metadata_differences"]:
            if item["kind"] == "progress":
                left, right = item["continuous"].copy(), item["resumed"].copy()
                left.pop("constraints_current", None)
                right.pop("constraints_current", None)
                if left == right:
                    continue
            retained.append(item)
        contract["raw_metadata_differences"] = retained
    return value


def compact_periodic(record):
    return compact(comparable_periodic(record))
