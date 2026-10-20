import io
import zipfile

import pytest

from changeradar.intake import IntakeError, scan_directory, scan_zip


def make_zip(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


def test_scan_directory_filters_dependencies_and_non_python(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "ok.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "readme.txt").write_text("ignore")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "bad.py").write_text("ignored = True")
    assert [(item.path, item.content) for item in scan_directory(tmp_path)] == [
        ("pkg/ok.py", "x = 1\n")
    ]


def test_scan_directory_skips_symlink_files(tmp_path):
    target = tmp_path / "actual.py"
    target.write_text("safe = True")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert [item.path for item in scan_directory(tmp_path)] == ["actual.py"]


def test_scan_zip_reads_without_extracting_and_filters_ignored_paths():
    sources = scan_zip(
        make_zip(
            [
                ("project/app.py", "def main(): pass"),
                ("project/.venv/pkg.py", "ignored = True"),
                ("project/image.png", b"\x00\xff"),
            ]
        )
    )
    assert [(item.path, item.content) for item in sources] == [
        ("project/app.py", "def main(): pass")
    ]


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "a\\..\\escape.py"])
def test_scan_zip_rejects_unsafe_paths(name):
    with pytest.raises(IntakeError, match="Unsafe ZIP path"):
        scan_zip(make_zip([(name, "x = 1")]))


def test_scan_zip_rejects_symlink_entries():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        info = zipfile.ZipInfo("link.py")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "target.py")
    with pytest.raises(IntakeError, match="symlinks"):
        scan_zip(output.getvalue())


def test_scan_zip_rejects_bad_archive():
    with pytest.raises(IntakeError, match="Invalid ZIP"):
        scan_zip(b"not zip")
