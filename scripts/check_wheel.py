import argparse
import compileall
import email
import importlib
import pkgutil
import re
import zipfile
from pathlib import Path


def archive(wheel, root):
    root = Path(root).resolve()
    source = {p.relative_to(root).as_posix() for p in (root / "ratchet").rglob("*.py")}
    with zipfile.ZipFile(wheel) as zf:
        packaged = {n for n in zf.namelist() if n.startswith("ratchet/") and n.endswith(".py")}
        assert packaged == source, f"wheel/source Python mismatch: {packaged ^ source}"
        metadata_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(zf.read(metadata_name))
    assert metadata["Name"] == "ratchet"
    version = metadata["Version"]
    assert version == "0.1.0"
    init_text = (root / "ratchet" / "__init__.py").read_text()
    assert re.search(r'__version__\s*=\s*["\']([^"\']+)', init_text).group(1) == version
    python_spec = {part.strip() for part in metadata["Requires-Python"].split(",")}
    assert python_spec == {">=3.12", "<3.15"}
    requires = metadata.get_all("Requires-Dist", [])
    assert any(value.lower().startswith("pyyaml") for value in requires)
    assert any(value.lower().startswith("pytest>=8") and "extra" in value.lower()
               and "dev" in value.lower() for value in requires)
    assert any(value.lower().startswith("build>=1") and "extra" in value.lower()
               and "dev" in value.lower() for value in requires)
    print(f"archive ok: {len(source)} Python modules")


def installed(prefix, expected_core=None):
    prefix = Path(prefix).resolve()
    import ratchet
    import yaml
    ratchet_root = Path(ratchet.__file__).resolve().parent
    yaml_root = Path(yaml.__file__).resolve()
    assert ratchet_root.is_relative_to(prefix), (ratchet_root, prefix)
    assert yaml_root.is_relative_to(prefix), (yaml_root, prefix)
    modules = [m.name for m in pkgutil.walk_packages(ratchet.__path__, ratchet.__name__ + ".")]
    for name in modules:
        importlib.import_module(name)
    assert compileall.compile_dir(ratchet_root, quiet=1)
    if expected_core is not None:
        from ratchet.regime import _core_fingerprint
        assert _core_fingerprint() == expected_core
    print(f"installed ok: {ratchet_root}; {len(modules) + 1} modules")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    archive_parser = sub.add_parser("archive")
    archive_parser.add_argument("wheel")
    archive_parser.add_argument("root")
    installed_parser = sub.add_parser("installed")
    installed_parser.add_argument("prefix")
    installed_parser.add_argument("--expected-core")
    args = parser.parse_args()
    if args.mode == "archive":
        archive(args.wheel, args.root)
    else:
        installed(args.prefix, args.expected_core)


if __name__ == "__main__":
    main()
