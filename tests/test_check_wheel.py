import zipfile

import pytest

from scripts.check_wheel import archive


def _write_wheel(path, root, *, changed_path):
    metadata = """\
Metadata-Version: 2.4
Name: ratchet
Version: 0.1.0
Requires-Python: >=3.12,<3.15
Requires-Dist: pyyaml
Provides-Extra: dev
Requires-Dist: pytest>=8; extra == "dev"
Requires-Dist: build>=1; extra == "dev"
"""
    with zipfile.ZipFile(path, "w") as zf:
        for source_path in (root / "ratchet").rglob("*.py"):
            relative = source_path.relative_to(root).as_posix()
            content = source_path.read_bytes()
            if relative == changed_path:
                content += b"\n# wheel-only change\n"
            zf.writestr(relative, content)
        zf.writestr("ratchet-0.1.0.dist-info/METADATA", metadata)


def test_archive_rejects_changed_python_bytes_with_identical_paths(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    source_root = root / "ratchet"
    source_root.mkdir()
    (source_root / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (source_root / "module.py").write_text("VALUE = 1\n")
    wheel = tmp_path / "ratchet-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, root, changed_path="ratchet/module.py")

    with pytest.raises(AssertionError, match=r"changed=.*ratchet/module\.py"):
        archive(wheel, root)
