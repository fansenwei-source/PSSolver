"""Run and report every D-0058 leg even when an ordinary check fails."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


BASE = "7267dcd00d58c11069a1ee205c181fa3f62ba745"


def compare_imports(root, basis, output):
    from qualification.s25_import_scan import scan

    old = scan(basis, output / "imports_basis.json", actual=True)
    new = scan(root, output / "imports_candidate.json", actual=True)

    def actual(row, tree):
        return {
            name: (item["exit_code"], item["stderr"].replace(str(tree), "<tree>"))
            for name, item in row["actual"].items()
            if item["exit_code"]
        }

    old_actual, new_actual = actual(old, basis), actual(new, root)
    actual_changes = sorted(
        k for k in old_actual.keys() | new_actual.keys() if old_actual.get(k) != new_actual.get(k)
    )
    common = old["modules"].keys() & new["modules"].keys()
    static_changes = sorted(k for k in common if old["modules"][k] != new["modules"][k])
    added = new["modules"].keys() - old["modules"].keys()
    # Existing modules are compared exactly. New modules must import and have no
    # unresolved project references; no new exemptions are inferred from a failure.
    invalid = sorted(k for k in added if new["modules"][k]["errors"])
    removed = sorted(old["modules"].keys() - new["modules"].keys())
    return {
        "actual": len(new["actual"]),
        "inherited_actual_failures": len(old_actual),
        "changed_actual_failures": actual_changes,
        "static_changes": static_changes,
        "added": sorted(added),
        "removed": removed,
        "invalid_added": invalid,
        "passed": not (actual_changes or static_changes or invalid or removed),
    }


def run(root, basis, output, suite):
    sys.path.insert(0, str(root))
    from qualification.s29_preflight import verify

    def git(repo, *args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    assert git(basis, "rev-parse", "HEAD") == BASE
    assert not git(root, "remote") and not git(basis, "remote")
    assert git(root, "rev-parse", "--is-shallow-repository") == "false"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    report = {
        "head": git(root, "rev-parse", "HEAD"),
        "basis": BASE,
        "before": git(root, "status", "--porcelain", "--ignored"),
    }
    assert not report["before"] and not git(root, "rev-list", "--merges", BASE + "..HEAD")

    def save():
        (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def execute(label, args):
        args = list(map(str, args))
        start = time.time()
        with (output / f"{label}.stdout").open("xb") as out:
            with (output / f"{label}.stderr").open("xb") as err:
                process = subprocess.run(
                    args, cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False
                )
        report[label] = dict(argv=args, exit=process.returncode, started=start, ended=time.time())
        save()

    def inspect(label, function):
        try:
            report[label] = function()
        except Exception:
            report[label] = {"passed": False, "error": traceback.format_exc()}
        save()

    py = [sys.executable, "-I", "-B"]
    for label in ("collection", "full"):
        execute(label, [*py, suite, root, output / f"{label}.json", label])
    inspect("imports", lambda: compare_imports(root, basis, output))
    inspect("classification", lambda: verify(root, output / "collection.json"))
    execute(
        "mechanical",
        [
            *py,
            root / "qualification/s29_guard.py",
            "final",
            root,
            root / "qualification/s29_authority.json",
            output / "guard.json",
        ],
    )
    execute("engineering", ["bash", "tools/check.sh", "--base", BASE])

    def outcomes():
        collect, full = (
            json.loads((output / f"{label}.json").read_bytes()) for label in ("collection", "full")
        )
        matched = full["nodeids"] == collect["nodeids"] and sorted(full["deselected"]) == sorted(
            collect["deselected"]
        )
        failures = [
            {"node": name, **item}
            for name, rows in full["outcomes"].items()
            for item in rows
            if item["outcome"] != "passed"
        ]
        passed = sum(
            all(row["outcome"] == "passed" for row in rows) for rows in full["outcomes"].values()
        )
        errors = collect["collection_errors"] + full["collection_errors"]
        return {
            "passed": matched
            and not failures
            and not errors
            and passed == len(collect["nodeids"])
            and not full["cuda_initialized"],
            "same_ids": matched,
            "count_passed": passed,
            "failures": failures,
            "collection_errors": errors,
            "deselected": collect["deselected"],
        }

    inspect("outcomes", outcomes)
    report["after"] = git(root, "status", "--porcelain", "--ignored")
    report["passed"] = (
        not report["after"]
        and all(
            report[key]["exit"] == 0 for key in ("collection", "full", "mechanical", "engineering")
        )
        and all(report[key]["passed"] for key in ("imports", "classification", "outcomes"))
    )
    save()
    return int(not report["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "basis", "output", "suite"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    return run(args.root, args.basis, args.output, args.suite)


if __name__ == "__main__":
    raise SystemExit(main())
