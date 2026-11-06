"""Exact reference comparisons, with explicit sampling-phase diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def compact(record: dict) -> dict:
    """Keep per-step field/metadata hashes; identities are compared in raw runs.

    Output paths belong to configuration identity, so portable stored baselines
    exclude *only* the identity object. No tensor, observation, progress or solver
    flag is normalized. The full three-way raw comparison includes identities.
    """
    result = {"scene": record["scene"], "disposition": record["disposition"]}
    if record["disposition"] == "rejected":
        result.update({key: record[key] for key in ("exception_type", "message")})
        return result
    trace = record["trace"]
    result["steps"] = [digest(row) for row in trace["steps"]]
    result["restarts"] = {
        key: [digest(row) for row in rows] for key, rows in trace["restarts"].items()
    }
    result["restart_contracts"] = trace["restart_contracts"]
    if "vjp" in record:
        result["vjp"] = digest(
            {key: value for key, value in record["vjp"].items() if key != "identity"}
        )
    return result


def require_finite(value):
    if isinstance(value, dict):
        if "sha256" in value and "finite" in value and value["finite"] is not True:
            raise AssertionError("Nonfinite tensor in S-23 trajectory")
        for child in value.values():
            require_finite(child)
    elif isinstance(value, list):
        for child in value:
            require_finite(child)


def validate_sampling(record: dict) -> dict:
    require_finite(record)
    if record["disposition"] == "rejected":
        return {}
    scene = record["scene"]
    counts = {"constraint_restore_flag": 0, "invalidated_constructor_cache": 0}
    for start, contract in record["trace"]["restart_contracts"].items():
        assert contract["field_bytes_and_clocks_equal"] and not contract["differences"]
        for row in contract["raw_metadata_differences"]:
            assert row["step"] == int(start), row
            left, right = row["continuous"], row["resumed"]
            if row["kind"] == "progress":
                assert scene["family"] in {"periodic", "plane"}
                assert row["phase"] == "after_checkpoint"
                assert left["constraints_current"] is False and right["constraints_current"] is True
                assert {**left, "constraints_current": True} == right
                counts["constraint_restore_flag"] += 1
            elif row["kind"] == "solver":
                assert scene["family"] == "channel"
                assert scene["options"].get("fixed_iterations", 12) is None
                assert left["operator_action_matches_guess"] is True
                assert right["operator_action_matches_guess"] is False
                counts["invalidated_constructor_cache"] += 1
            else:
                raise AssertionError(f"Unclassified metadata difference: {row}")
    return counts


def compare(roots: list[Path]) -> dict:
    assert len(roots) == 3
    environments = [json.loads((root / "environment.json").read_text()) for root in roots]
    assert (
        environments[0]["fingerprint"]
        == environments[1]["fingerprint"]
        == environments[2]["fingerprint"]
    )
    sets = [
        {path.name for path in root.glob("*.json")} - {"environment.json", "origins.json"}
        for root in roots
    ]
    assert sets[0] == sets[1] == sets[2] and "STOPPED.json" not in sets[0]
    cases, sampling = {}, {}
    for name in sorted(sets[0]):
        records = [json.loads((root / name).read_text()) for root in roots]
        assert records[0] == records[1] == records[2], name
        sampling[name] = validate_sampling(records[0])
        cases[name] = compact(records[0])
    p5 = {}
    for name in sorted(sets[0]):
        if not name.endswith("_unobserved.json"):
            continue
        observed = json.loads((roots[0] / name.replace("_unobserved", "")).read_text())["trace"][
            "steps"
        ]
        unobserved = json.loads((roots[0] / name).read_text())["trace"]["steps"]
        # Compare after constraint synchronization, before explicit RHS; the
        # unsynchronized hydrodynamic cache is not a physical-time observation.
        for left, right in zip(observed[:-1], unobserved[:-1], strict=True):
            assert (
                left["before_explicit_rhs"]["tensors"] == right["before_explicit_rhs"]["tensors"]
            ), name
        p5[name] = {"steps": len(observed) - 1, "pre_rhs_field_bytes_equal": True}
    return {
        "schema": 1,
        "fingerprint": environments[0]["fingerprint"],
        "cases": cases,
        "sampling_audit": sampling,
        "p5": p5,
        "three_way_raw_equality_including_identities": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(compare(args.roots), stream, sort_keys=True, indent=2)
        stream.write("\n")
