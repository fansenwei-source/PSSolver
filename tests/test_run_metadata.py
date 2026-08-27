import pytest

from pssolver import prepare_new_run_directory


def test_prepare_new_run_directory_creates_missing_directory(tmp_path):
    output = tmp_path / "nested" / "run"

    result = prepare_new_run_directory(output)

    assert result == output
    assert output.is_dir()


def test_prepare_new_run_directory_accepts_existing_empty_directory(tmp_path):
    output = tmp_path / "run"
    output.mkdir()

    assert prepare_new_run_directory(output) == output


def test_prepare_new_run_directory_rejects_nonempty_directory(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    existing = output / "Q_0.npy"
    existing.write_bytes(b"existing simulation data")

    with pytest.raises(FileExistsError, match="nonempty directory"):
        prepare_new_run_directory(output)

    assert existing.read_bytes() == b"existing simulation data"


def test_prepare_new_run_directory_rejects_existing_file(tmp_path):
    output = tmp_path / "run"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not a directory"):
        prepare_new_run_directory(output)

    assert output.read_text(encoding="utf-8") == "not a directory"
