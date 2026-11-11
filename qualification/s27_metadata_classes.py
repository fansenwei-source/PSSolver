"""R-0092: classify every leaf of Plane (and Periodic) metadata.json into the S-25 §4.4 classes.

Usage: python3 -I -B s27_metadata_classes.py <rules.json> <metadata.json>... > report
Rules are ordered; the first matching pattern wins (fnmatch on the JSON pointer, '/**' = subtree).
Any leaf matching no rule is UNCLASSIFIED and makes the exit code 1 (fail closed).
With exactly two files A B it also prints the differing leaves and their classes.
"""

import fnmatch
import json
import sys


def main():
    rules = json.load(open(sys.argv[1]))
    files = sys.argv[2:]

    def leaves(x, p=""):
        if isinstance(x, dict):
            if not x:
                yield p, {}
            for k, v in x.items():
                yield from leaves(v, f"{p}/{k}")
        elif isinstance(x, list):
            if not x:
                yield p, []
            for i, v in enumerate(x):
                yield from leaves(v, f"{p}/{i}")
        else:
            yield p, x

    def classify(pointer):
        for rule in rules["rules"]:
            pat = rule["pattern"]
            if pat.endswith("/**"):
                base = pat[:-3]
                if (
                    pointer == base
                    or pointer.startswith(base + "/")
                    or fnmatch.fnmatchcase(pointer, base + "/*")
                ):
                    return rule["class"]
            elif fnmatch.fnmatchcase(pointer, pat):
                return rule["class"]
        return "UNCLASSIFIED"

    bad = 0
    data = {}
    for f in files:
        d = dict(leaves(json.load(open(f))))
        data[f] = d
        counts = {}
        for p in d:
            c = classify(p)
            counts[c] = counts.get(c, 0) + 1
            if c == "UNCLASSIFIED":
                bad += 1
                print("UNCLASSIFIED", f, p)
        print(f, len(d), counts)
    if len(files) == 2:
        a, b = (data[f] for f in files)
        for p in sorted(set(a) | set(b)):
            if a.get(p, "<missing>") != b.get(p, "<missing>"):
                print(
                    "DIFF",
                    classify(p),
                    p,
                    json.dumps(a.get(p, "<missing>"))[:80],
                    "|",
                    json.dumps(b.get(p, "<missing>"))[:80],
                )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
