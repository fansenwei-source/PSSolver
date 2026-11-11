"""Plane comparison: precisely the two S-27 section 9 exclusions, no tolerance."""

from copy import deepcopy

from qualification.identity.analyze import compact


def comparable_plane(record):
    """Keep raw records intact; exclude implementation class and scheduling cache only."""
    value = deepcopy(record)
    if value["scene"]["family"] != "plane" or value["disposition"] != "accepted":
        return value
    for identity in (value.get("identity"), value.get("trace", {}).get("identity")):
        if identity is not None:
            identity["runtime"].pop("adapter")
    trace = value["trace"]
    for rows in [trace["steps"], *trace["restarts"].values()]:
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


def compact_plane(record):
    return compact(comparable_plane(record))
