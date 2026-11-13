"""Pin the accepted S-03 record order without editing a historical checkout."""

import hashlib
import importlib
import re
from pathlib import Path


RESOLVER = "tests/v0_3_s03_successor.py"
ACCEPTED_RESOLVERS = frozenset(
    {
        "0368f3030517b1971984608297afbdb05647d151a09ea154c6eafe59e0bcdfe0",
        "87ad03ab32aa5f0fb313d1b85e80dc1a88820737299c2d22418f86a6e4a9e7c4",
    }
)


def pin(checkout: Path) -> None:
    """Verify the historical resolver, then fix only its record traversal."""
    root = checkout.resolve()
    path = (root / RESOLVER).resolve()
    if path != root / RESOLVER or not path.is_file():
        raise RuntimeError(f"S-03 resolver not from the historical checkout: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest not in ACCEPTED_RESOLVERS:
        raise RuntimeError(f"unexpected S-03 resolver bytes: {digest}")
    resolver = importlib.import_module("tests.v0_3_s03_successor")
    if Path(resolver.__file__).resolve() != path:
        raise RuntimeError(f"S-03 resolver import does not match the verified file: {path}")

    def ordered_records() -> list[Path]:
        successors = sorted(
            (resolver.ROOT / "notes").glob(resolver.SUCCESSOR_PATTERN),
            key=lambda item: (int(re.search(r"_successor_(\d+)", item.name).group(1)), item.name),
        )
        return [resolver.BASE_RECORD, *successors]

    resolver._records = ordered_records
