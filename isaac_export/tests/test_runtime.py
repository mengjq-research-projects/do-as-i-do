from importlib.metadata import PackageNotFoundError

from isaac_export.runtime import isaac_sim_version


def test_version_uses_package_metadata(monkeypatch):
    monkeypatch.setattr(
        "isaac_export.runtime.importlib.metadata.version",
        lambda name: "6.0.1.0",
    )

    assert isaac_sim_version() == "6.0.1.0"


def test_version_reads_standalone_version_file(monkeypatch, tmp_path):
    def missing_metadata(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(
        "isaac_export.runtime.importlib.metadata.version", missing_metadata
    )
    monkeypatch.setenv("ISAAC_PATH", str(tmp_path))
    (tmp_path / "VERSION").write_text(
        "5.1.0-rc.19+release.26219.9c81211b.gl\n", encoding="utf-8"
    )

    assert isaac_sim_version() == "5.1.0-rc.19+release.26219.9c81211b.gl"


def test_version_is_unknown_without_metadata_or_standalone_root(monkeypatch):
    def missing_metadata(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(
        "isaac_export.runtime.importlib.metadata.version", missing_metadata
    )
    monkeypatch.delenv("ISAAC_PATH", raising=False)

    assert isaac_sim_version() == "unknown"
