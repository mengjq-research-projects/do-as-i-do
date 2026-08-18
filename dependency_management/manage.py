#!/usr/bin/env python3
"""Manage versioned external artifacts and portable runtime environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RELEASE = "2026.08-a100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "sync-assets",
            "verify-assets",
            "prepare-runtime",
            "verify-runtime",
            "pack-conda",
            "verify-conda-packs",
            "install-conda",
            "verify-installed-conda",
            "sync-uv-cache",
            "verify-uv-cache",
            "install-venvs",
            "verify-installed-venvs",
            "verify-submodules",
        ),
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument(
        "--root",
        type=Path,
        help="Overrides DO_AS_I_DO_PACKAGE_ROOT and the manifest artifact_root.",
    )
    parser.add_argument(
        "--environment",
        action="append",
        help="Limit an environment command to one environment; may be repeated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild an existing Conda archive (never replaces installs).",
    )
    return parser.parse_args()


def load_release(release_id: str) -> tuple[dict, Path]:
    path = HERE / "releases" / f"{release_id}.json"
    if not path.is_file():
        raise SystemExit(f"Release manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def resolve_release_directory(args: argparse.Namespace, manifest: dict) -> Path:
    configured = args.root or os.environ.get("DO_AS_I_DO_PACKAGE_ROOT")
    root = Path(configured or manifest["artifact_root"]).expanduser().resolve()
    release = (root / "releases" / manifest["release_id"]).resolve()
    if root not in release.parents:
        raise SystemExit(f"Invalid release directory outside package root: {release}")
    release.mkdir(parents=True, exist_ok=True)
    return release


def artifact_entries(manifest: dict) -> list[dict]:
    entries = manifest.get("source_assets", [])
    missing = [
        entry.get("name", "<unnamed>") for entry in entries if "target" not in entry
    ]
    if missing:
        raise SystemExit("Artifacts missing target paths: " + ", ".join(missing))
    return entries


def target_path(release_directory: Path, entry: dict) -> Path:
    target = (release_directory / entry["target"]).resolve()
    if release_directory not in target.parents:
        raise SystemExit(f"Artifact target escapes release directory: {target}")
    return target


def release_relative_path(
    release_directory: Path,
    relative: str,
    label: str,
) -> Path:
    path = (release_directory / relative).resolve()
    if release_directory not in path.parents:
        raise SystemExit(f"{label} escapes release directory: {path}")
    return path


def sync_assets(manifest: dict, release_directory: Path) -> None:
    for entry in artifact_entries(manifest):
        source = Path(entry["path"]).expanduser().resolve()
        target = target_path(release_directory, entry)
        if not source.is_dir():
            raise SystemExit(f"Missing artifact source for {entry['name']}: {source}")
        target.mkdir(parents=True, exist_ok=True)
        print(f"Syncing {entry['name']}: {source} -> {target}", flush=True)
        subprocess.run(
            [
                "rsync",
                "-aH",
                "--no-owner",
                "--no-group",
                "--partial",
                "--info=progress2",
                f"{source}/",
                f"{target}/",
            ],
            check=True,
        )
    write_checksums(manifest, release_directory)


def iter_source_files(manifest: dict):
    for entry in artifact_entries(manifest):
        source = Path(entry["path"]).expanduser().resolve()
        target_prefix = Path(entry["target"])
        if not source.is_dir():
            raise SystemExit(f"Missing artifact source for {entry['name']}: {source}")
        for path in sorted(
            candidate for candidate in source.rglob("*") if candidate.is_file()
        ):
            yield path, target_prefix / path.relative_to(source)


def iter_artifact_files(manifest: dict, release_directory: Path):
    for entry in artifact_entries(manifest):
        target = target_path(release_directory, entry)
        if not target.is_dir():
            raise SystemExit(f"Missing artifact target for {entry['name']}: {target}")
        for path in sorted(
            candidate for candidate in target.rglob("*") if candidate.is_file()
        ):
            yield path, path.relative_to(release_directory)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_path(release_directory: Path) -> Path:
    return release_directory / "manifests" / "assets.sha256"


def write_checksums(manifest: dict, release_directory: Path) -> None:
    output = checksum_path(release_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".sha256.tmp")
    count = 0
    total = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for path, relative in iter_source_files(manifest):
            stream.write(f"{sha256(path)}  {relative.as_posix()}\n")
            count += 1
            total += path.stat().st_size
    temporary.replace(output)
    print(f"Wrote {count} source checksums for {total} bytes to {output}")


def read_checksums(path: Path) -> dict[Path, str]:
    if not path.is_file():
        raise SystemExit(f"Checksum manifest does not exist: {path}")
    expected: dict[Path, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise SystemExit(f"Invalid checksum line {line_number} in {path}")
        expected[Path(relative)] = digest
    return expected


def verify_assets(manifest: dict, release_directory: Path) -> None:
    expected = read_checksums(checksum_path(release_directory))
    actual_paths = {
        relative for _, relative in iter_artifact_files(manifest, release_directory)
    }
    expected_paths = set(expected)
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    failures = []
    for relative in sorted(expected_paths & actual_paths):
        path = release_directory / relative
        actual_digest = sha256(path)
        if actual_digest != expected[relative]:
            failures.append(relative)
    if missing or unexpected or failures:
        for label, paths in (
            ("missing", missing),
            ("unexpected", unexpected),
            ("checksum mismatch", failures),
        ):
            for path in paths:
                print(f"{label}: {path}")
        raise SystemExit("Artifact verification failed.")
    print(f"Verified {len(expected)} artifact files: PASS")


def replace_generated_symlink(link: Path, target: Path) -> None:
    """Create a release-local generated link without replacing user files."""
    target = target.resolve()
    if not target.is_file():
        raise SystemExit(f"Runtime link target does not exist: {target}")
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise SystemExit(
            f"Refusing to replace non-symlink runtime artifact: {link}"
        )
    link.symlink_to(target)


def copy_runtime_configs(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"Runtime config source does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    configs = sorted(source.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"No YAML runtime configs found in: {source}")
    for config in configs:
        shutil.copy2(config, destination / config.name)


def sam3d_weight_directory(release_directory: Path) -> Path:
    return (
        release_directory
        / "assets/weights/reconstruction/sam3d_shared/hf"
    )


def prepare_runtime(release_directory: Path) -> None:
    """Build small config/link trees consumed by the reconstruction scripts."""
    project_root = HERE.parent
    runtime_root = release_directory / "runtime"
    # The upstream sam-3d-objects checkout ignores checkpoints/hf entirely.
    # Fast-SAM3D tracks the same canonical config set, so use it to build both
    # runtime trees instead of depending on a locally downloaded HF snapshot.
    tracked_config_source = (
        project_root / "reconstruction/modules/Fast-SAM3D/checkpoints/hf"
    )
    sources = {
        "sam3d": tracked_config_source,
        "fast-sam3d": tracked_config_source,
    }
    weight_directory = sam3d_weight_directory(release_directory)
    if not weight_directory.is_dir():
        raise SystemExit(
            "SAM3D weights are missing from the release. Run sync-assets first: "
            f"{weight_directory}"
        )
    weights = sorted(weight_directory.glob("*"))
    weights = [path for path in weights if path.is_file()]
    if not weights:
        raise SystemExit(f"No SAM3D weights found in: {weight_directory}")

    for name, config_source in sources.items():
        destination = runtime_root / name / "checkpoints/hf"
        copy_runtime_configs(config_source, destination)
        for weight in weights:
            replace_generated_symlink(destination / weight.name, weight)

    marker = runtime_root / ".managed-by-do-as-i-do"
    marker.write_text(
        f"release={release_directory.name}\n",
        encoding="utf-8",
    )
    print(f"Prepared runtime resource links in {runtime_root}")


def required_runtime_paths(release_directory: Path) -> list[Path]:
    reconstruction = release_directory / "assets/weights/reconstruction"
    hawor = release_directory / "assets/weights/hawor"
    metric3d = release_directory / "assets/weights/metric3d"
    deployment = release_directory / "assets/third-party/deployment-assets"
    required = [
        reconstruction / "sam3/sam3.pt",
        reconstruction / "moge-vitl/model.pt",
        reconstruction / "geocalib/pinhole.tar",
        reconstruction / "tapnet/bootstapir_checkpoint_v2.pt",
        hawor / "hawor/checkpoints/hawor.ckpt",
        hawor / "hawor/checkpoints/infiller.pt",
        hawor / "external/detector.pt",
        hawor / "external/droid.pth",
        metric3d / "metric_depth_vit_large_800k.pth",
        deployment / "dinov2/hubconf.py",
    ]
    for runtime_name in ("sam3d", "fast-sam3d"):
        checkpoint_dir = runtime_name + "/checkpoints/hf"
        required.append(release_directory / "runtime" / checkpoint_dir / "pipeline.yaml")
        for weight in sorted(sam3d_weight_directory(release_directory).glob("*")):
            if weight.is_file():
                required.append(
                    release_directory / "runtime" / checkpoint_dir / weight.name
                )
    return required


def verify_runtime(release_directory: Path) -> None:
    paths = required_runtime_paths(release_directory)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"missing: {path}")
        raise SystemExit("Runtime resource verification failed.")
    print(f"Verified {len(paths)} runtime paths: PASS")


def conda_environment_entries(manifest: dict, selected: list[str] | None) -> list[dict]:
    entries = [
        entry
        for entry in manifest.get("source_environments", [])
        if entry.get("kind") == "conda" and entry.get("pack")
    ]
    known = {entry["name"] for entry in entries}
    requested = set(selected or known)
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(
            "Unknown or non-Conda environment(s): " + ", ".join(unknown)
        )
    return [entry for entry in entries if entry["name"] in requested]


def conda_pack_executable() -> str:
    configured = os.environ.get("CONDA_PACK_EXE")
    candidates = [
        configured,
        shutil.which("conda-pack"),
        str(Path(sys.executable).parent / "conda-pack"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "conda-pack is required only to build archives. Install the pinned "
        "offline tool wheel described in dependency_management/README.md."
    )


def conda_pack_metadata_path(release_directory: Path) -> Path:
    return release_directory / "manifests/conda-packs.json"


def load_conda_pack_metadata(release_directory: Path) -> dict:
    path = conda_pack_metadata_path(release_directory)
    if not path.is_file():
        return {"schema_version": 1, "environments": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_conda_pack_metadata(release_directory: Path, metadata: dict) -> None:
    path = conda_pack_metadata_path(release_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pack_conda_environments(
    manifest: dict,
    release_directory: Path,
    selected: list[str] | None,
    force: bool,
) -> None:
    packer = conda_pack_executable()
    metadata = load_conda_pack_metadata(release_directory)
    environments = metadata.setdefault("environments", {})
    for entry in conda_environment_entries(manifest, selected):
        source = Path(entry["path"]).expanduser().resolve()
        archive = release_relative_path(
            release_directory, entry["pack"], "Conda archive"
        )
        if not source.joinpath("bin/python").is_file():
            raise SystemExit(f"Invalid Conda environment source: {source}")
        if archive.is_file() and not force:
            print(f"Keeping existing archive (use --force to rebuild): {archive}")
            continue
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_name(f".{archive.name}.packing")
        if temporary.exists():
            temporary.unlink()
        print(f"Packing {entry['name']}: {source} -> {archive}", flush=True)
        subprocess.run(
            [
                packer,
                "--prefix",
                str(source),
                "--output",
                str(temporary),
                "--format",
                "tar.zst",
                "--compress-level",
                "3",
                "--n-threads",
                "-1",
                "--ignore-editable-packages",
                "--ignore-missing-files",
                "--force",
            ],
            check=True,
        )
        temporary.replace(archive)
        digest = sha256(archive)
        environments[entry["name"]] = {
            "archive": entry["pack"],
            "bytes": archive.stat().st_size,
            "python": entry["python"],
            "sha256": digest,
            "source_prefix": str(source),
        }
        metadata["generated_at"] = datetime.now(timezone.utc).isoformat()
        write_conda_pack_metadata(release_directory, metadata)
        print(f"Packed {entry['name']}: {archive.stat().st_size} bytes, sha256={digest}")


def verified_conda_archive(
    release_directory: Path,
    entry: dict,
    metadata: dict,
) -> tuple[Path, str]:
    recorded = metadata.get("environments", {}).get(entry["name"])
    if not recorded:
        if not entry.get("pack_sha256"):
            raise SystemExit(
                f"No pack metadata for Conda environment: {entry['name']}"
            )
        recorded = {
            "archive": entry["pack"],
            "bytes": entry.get("pack_bytes"),
            "sha256": entry["pack_sha256"],
        }
    tracked_digest = entry.get("pack_sha256")
    if tracked_digest and recorded["sha256"] != tracked_digest:
        raise SystemExit(
            f"Generated and tracked checksums disagree for {entry['name']}"
        )
    archive = release_relative_path(
        release_directory, recorded["archive"], "Conda archive"
    )
    if not archive.is_file():
        raise SystemExit(f"Missing Conda archive: {archive}")
    digest = sha256(archive)
    if digest != recorded["sha256"]:
        raise SystemExit(f"Checksum mismatch for Conda archive: {archive}")
    tracked_bytes = entry.get("pack_bytes")
    if tracked_bytes and archive.stat().st_size != tracked_bytes:
        raise SystemExit(f"Size mismatch for Conda archive: {archive}")
    return archive, digest


def verify_conda_packs(
    manifest: dict,
    release_directory: Path,
    selected: list[str] | None,
) -> None:
    metadata = load_conda_pack_metadata(release_directory)
    entries = conda_environment_entries(manifest, selected)
    for entry in entries:
        archive, digest = verified_conda_archive(
            release_directory, entry, metadata
        )
        print(f"{entry['name']}: PASS ({archive.stat().st_size} bytes, {digest})")
    print(f"Verified {len(entries)} Conda environment packs: PASS")


def installed_conda_prefix(release_directory: Path, entry: dict) -> Path:
    return release_relative_path(
        release_directory, entry["install_target"], "Conda install target"
    )


def install_conda_environments(
    manifest: dict,
    release_directory: Path,
    selected: list[str] | None,
) -> None:
    metadata = load_conda_pack_metadata(release_directory)
    for entry in conda_environment_entries(manifest, selected):
        target = installed_conda_prefix(release_directory, entry)
        marker = target / ".do-as-i-do-env.json"
        tracked_digest = entry.get("pack_sha256")
        tracked_bytes = entry.get("pack_bytes")
        archive = release_relative_path(
            release_directory, entry["pack"], "Conda archive"
        )
        if marker.is_file() and target.joinpath("bin/python").is_file():
            installed = json.loads(marker.read_text(encoding="utf-8"))
            if (
                tracked_digest
                and installed.get("archive_sha256") == tracked_digest
                and archive.is_file()
                and (not tracked_bytes or archive.stat().st_size == tracked_bytes)
            ):
                print(f"Already installed: {entry['name']} -> {target}")
                continue
        if target.exists():
            raise SystemExit(
                "Refusing to replace an existing unverified environment: "
                f"{target}"
            )
        archive, digest = verified_conda_archive(
            release_directory, entry, metadata
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.installing-{os.getpid()}")
        if temporary.exists():
            raise SystemExit(f"Temporary install path already exists: {temporary}")
        temporary.mkdir()
        print(f"Installing {entry['name']}: {archive} -> {target}", flush=True)
        try:
            subprocess.run(
                ["tar", "--zstd", "-xf", str(archive), "-C", str(temporary)],
                check=True,
            )
            if not temporary.joinpath("bin/conda-unpack").is_file():
                raise RuntimeError(f"conda-unpack is missing from {temporary}")
            temporary.rename(target)
            subprocess.run(
                [
                    str(target / "bin/python"),
                    str(target / "bin/conda-unpack"),
                ],
                check=True,
            )
            marker_data = {
                "archive": str(archive.relative_to(release_directory)),
                "archive_sha256": digest,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "name": entry["name"],
            }
            marker.write_text(
                json.dumps(marker_data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except BaseException:
            if target.exists() and not temporary.exists():
                target.rename(temporary)
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        print(f"Installed {entry['name']} -> {target}")


def verify_installed_conda(
    manifest: dict,
    release_directory: Path,
    selected: list[str] | None,
) -> None:
    metadata = load_conda_pack_metadata(release_directory)
    entries = conda_environment_entries(manifest, selected)
    for entry in entries:
        recorded = metadata.get("environments", {}).get(entry["name"], {})
        digest = entry.get("pack_sha256") or recorded.get("sha256")
        if not digest:
            raise SystemExit(
                f"No expected archive checksum for {entry['name']}"
            )
        if entry.get("pack_sha256") and recorded.get("sha256") not in (
            None,
            entry["pack_sha256"],
        ):
            raise SystemExit(
                f"Generated and tracked checksums disagree for {entry['name']}"
            )
        target = installed_conda_prefix(release_directory, entry)
        marker = target / ".do-as-i-do-env.json"
        python = target / "bin/python"
        if not marker.is_file() or not python.is_file():
            raise SystemExit(f"Conda environment is not installed: {target}")
        installed = json.loads(marker.read_text(encoding="utf-8"))
        if installed.get("archive_sha256") != digest:
            raise SystemExit(f"Installed environment has stale metadata: {target}")
        version_result = subprocess.run(
            [str(python), "-c", "import sys; print(sys.version.split()[0])"],
            check=True,
            capture_output=True,
            text=True,
        )
        actual_python = version_result.stdout.strip()
        if actual_python != entry["python"]:
            raise SystemExit(
                f"Python mismatch for {entry['name']}: "
                f"expected {entry['python']}, found {actual_python}"
            )
        smoke_code = [
            "import importlib",
            *[
                f"importlib.import_module({module!r})"
                for module in entry.get("smoke_imports", [])
            ],
        ]
        if entry.get("cuda_required"):
            smoke_code.extend(
                [
                    "import torch",
                    "assert torch.cuda.is_available(), 'CUDA is unavailable'",
                    "print('cuda=' + str(torch.version.cuda))",
                    "print('capability=' + '.'.join(map(str, torch.cuda.get_device_capability())))",
                ]
            )
        smoke_result = subprocess.run(
            [str(python), "-c", ";".join(smoke_code)],
            capture_output=True,
            text=True,
        )
        if smoke_result.returncode:
            details = (smoke_result.stderr or smoke_result.stdout).strip()
            raise SystemExit(f"Smoke test failed for {entry['name']}: {details}")
        smoke_summary = ", ".join(
            line for line in smoke_result.stdout.splitlines() if line
        )
        print(
            f"{entry['name']}: PASS ({target}, Python {actual_python}; "
            f"{smoke_summary})"
        )
    print(f"Verified {len(entries)} installed Conda environments: PASS")


def venv_environment_entries(
    manifest: dict,
    selected: list[str] | None,
) -> list[dict]:
    entries = [
        entry
        for entry in manifest.get("source_environments", [])
        if entry.get("kind") == "venv" and entry.get("install_target")
    ]
    known = {entry["name"] for entry in entries}
    requested = set(selected or known)
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(
            "Unknown or non-venv environment(s): " + ", ".join(unknown)
        )
    return [entry for entry in entries if entry["name"] in requested]


def package_root_from_release(release_directory: Path) -> Path:
    # <package-root>/releases/<release-id>
    return release_directory.parents[1]


def python_tooling(manifest: dict) -> dict:
    tooling = manifest.get("python_tooling")
    required = {
        "uv_version",
        "uv_binary_source",
        "uv_binary_target",
        "uv_cache_source",
        "uv_cache_target",
    }
    missing = sorted(required - set(tooling or {}))
    if not tooling or missing:
        raise SystemExit(
            "Release manifest has incomplete python_tooling settings: "
            + ", ".join(missing if tooling else sorted(required))
        )
    return tooling


def uv_cache_directory(manifest: dict, release_directory: Path) -> Path:
    package_root = package_root_from_release(release_directory)
    relative = python_tooling(manifest)["uv_cache_target"]
    path = (package_root / relative).resolve()
    if package_root not in path.parents:
        raise SystemExit(f"uv cache target escapes package root: {path}")
    return path


def uv_binary(manifest: dict, release_directory: Path) -> Path:
    return release_relative_path(
        release_directory,
        python_tooling(manifest)["uv_binary_target"],
        "uv binary target",
    )


def project_input_hashes(entry: dict) -> dict[str, str]:
    project = (HERE.parent / entry["project"]).resolve()
    inputs = {}
    for filename in ("pyproject.toml", "uv.lock"):
        path = project / filename
        if not path.is_file():
            raise SystemExit(f"Missing {entry['name']} dependency input: {path}")
        inputs[filename] = sha256(path)
    return inputs


def uv_cache_marker_path(manifest: dict, release_directory: Path) -> Path:
    return uv_cache_directory(manifest, release_directory) / ".do-as-i-do-cache.json"


def sync_uv_cache(manifest: dict, release_directory: Path) -> None:
    tooling = python_tooling(manifest)
    source_binary = Path(tooling["uv_binary_source"]).expanduser().resolve()
    source_cache = Path(tooling["uv_cache_source"]).expanduser().resolve()
    target_binary = uv_binary(manifest, release_directory)
    target_cache = uv_cache_directory(manifest, release_directory)
    if not source_binary.is_file():
        raise SystemExit(f"uv binary source does not exist: {source_binary}")
    if not source_cache.is_dir():
        raise SystemExit(f"uv cache source does not exist: {source_cache}")

    target_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_binary, target_binary)
    target_binary.chmod(0o755)
    target_cache.mkdir(parents=True, exist_ok=True)
    print(f"Syncing uv cache: {source_cache} -> {target_cache}", flush=True)
    subprocess.run(
        [
            "rsync",
            "-aH",
            "--no-owner",
            "--no-group",
            "--partial",
            "--exclude=.tmp*",
            "--info=stats2",
            f"{source_cache}/",
            f"{target_cache}/",
        ],
        check=True,
    )
    marker = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_inputs": {
            entry["name"]: project_input_hashes(entry)
            for entry in venv_environment_entries(manifest, None)
        },
        "source": str(source_cache),
        "uv_version": tooling["uv_version"],
    }
    uv_cache_marker_path(manifest, release_directory).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Stored uv {tooling['uv_version']} and shared cache: PASS")


def verify_uv_cache(manifest: dict, release_directory: Path) -> None:
    tooling = python_tooling(manifest)
    binary = uv_binary(manifest, release_directory)
    cache = uv_cache_directory(manifest, release_directory)
    marker_path = uv_cache_marker_path(manifest, release_directory)
    required_cache_paths = ["archive-v0", "git-v0", "wheels-v6"]
    missing = [
        cache / name
        for name in required_cache_paths
        if not (cache / name).is_dir()
    ]
    if not binary.is_file() or not marker_path.is_file() or missing:
        for path in missing:
            print(f"missing: {path}")
        raise SystemExit(
            "Managed uv tool/cache is incomplete. Run sync-uv-cache on the "
            "validated builder first."
        )
    version = subprocess.run(
        [str(binary), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_version = tooling["uv_version"]
    if version != f"uv {expected_version} (x86_64-unknown-linux-gnu)":
        raise SystemExit(
            f"uv version mismatch: expected {expected_version}, found {version}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("uv_version") != expected_version:
        raise SystemExit(f"Stale uv cache marker: {marker_path}")
    recorded_inputs = marker.get("project_inputs", {})
    for entry in venv_environment_entries(manifest, None):
        if recorded_inputs.get(entry["name"]) != project_input_hashes(entry):
            raise SystemExit(
                f"uv cache predates the current {entry['name']} lock files; "
                "run sync-uv-cache on the validated builder."
            )
    print(f"uv {expected_version} and shared offline cache: PASS ({cache})")


def installed_venv_prefix(release_directory: Path, entry: dict) -> Path:
    return release_relative_path(
        release_directory, entry["install_target"], "venv install target"
    )


def venv_identity(manifest: dict, entry: dict) -> dict:
    identity = {
        "inputs": project_input_hashes(entry),
        "name": entry["name"],
        "project": str((HERE.parent / entry["project"]).resolve()),
        "python": entry["python"],
        "uv_version": python_tooling(manifest)["uv_version"],
    }
    if entry.get("isaac_sim"):
        identity["isaac_sim"] = entry["isaac_sim"]
    return identity


def venv_marker_path(target: Path) -> Path:
    return target / ".do-as-i-do-env.json"


def install_venv_environments(
    manifest: dict,
    release_directory: Path,
    selected: list[str] | None,
) -> None:
    verify_uv_cache(manifest, release_directory)
    binary = uv_binary(manifest, release_directory)
    cache = uv_cache_directory(manifest, release_directory)
    for entry in venv_environment_entries(manifest, selected):
        target = installed_venv_prefix(release_directory, entry)
        marker_path = venv_marker_path(target)
        identity = venv_identity(manifest, entry)
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("identity") == identity:
                print(f"Already installed: {entry['name']} -> {target}")
                continue
        if target.exists():
            raise SystemExit(
                f"Refusing to replace an existing unverified venv: {target}"
            )

        project = (HERE.parent / entry["project"]).resolve()
        base_python = release_relative_path(
            release_directory, entry["base_python"], "managed base Python"
        )
        if not base_python.is_file():
            raise SystemExit(
                f"Managed base Python is missing for {entry['name']}: {base_python}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "UV_CACHE_DIR": str(cache),
                "UV_OFFLINE": "1",
                "UV_PROJECT_ENVIRONMENT": str(target),
            }
        )
        sync_command = [
            str(binary),
            "sync",
            "--offline",
            "--frozen",
            "--project",
            str(project),
            "--python",
            str(base_python),
        ]
        for extra in entry.get("extras", []):
            sync_command.extend(["--extra", extra])
        print(f"Installing {entry['name']} offline -> {target}", flush=True)
        try:
            subprocess.run(sync_command, env=environment, check=True)
            if entry.get("isaac_sim"):
                wheelhouse = release_relative_path(
                    release_directory, entry["wheelhouse"], "Isaac wheelhouse"
                )
                if not wheelhouse.is_dir():
                    raise RuntimeError(f"Isaac wheelhouse is missing: {wheelhouse}")
                subprocess.run(
                    [
                        str(binary),
                        "pip",
                        "install",
                        "--offline",
                        "--no-index",
                        "--find-links",
                        str(wheelhouse),
                        "--python",
                        str(target / "bin/python"),
                        f"isaacsim[all,extscache]=={entry['isaac_sim']}",
                    ],
                    env=environment,
                    check=True,
                )
            marker_path.write_text(
                json.dumps(
                    {
                        "identity": identity,
                        "installed_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except BaseException:
            # The target did not exist before this function entered, so cleanup
            # cannot remove a user's pre-existing environment.
            shutil.rmtree(target, ignore_errors=True)
            raise
        print(f"Installed {entry['name']} -> {target}")


def verify_installed_venvs(
    manifest: dict,
    release_directory: Path,
    selected: list[str] | None,
) -> None:
    verify_uv_cache(manifest, release_directory)
    entries = venv_environment_entries(manifest, selected)
    for entry in entries:
        target = installed_venv_prefix(release_directory, entry)
        marker_path = venv_marker_path(target)
        python = target / "bin/python"
        if not marker_path.is_file() or not python.is_file():
            raise SystemExit(f"Managed venv is not installed: {target}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("identity") != venv_identity(manifest, entry):
            raise SystemExit(f"Managed venv has stale metadata: {target}")
        smoke_code = [
            "import importlib, importlib.metadata as metadata, sys",
            f"assert sys.version.split()[0] == {entry['python']!r}",
            *[
                f"importlib.import_module({module!r})"
                for module in entry.get("smoke_imports", [])
            ],
        ]
        for distribution, expected in entry.get(
            "smoke_distributions", {}
        ).items():
            smoke_code.append(
                f"assert metadata.version({distribution!r}) == {expected!r}"
            )
            smoke_code.append(
                f"print({distribution!r} + '=' + metadata.version({distribution!r}))"
            )
        if entry.get("cuda_required"):
            smoke_code.extend(
                [
                    "import torch",
                    "assert torch.cuda.is_available(), 'CUDA is unavailable'",
                    "print('cuda=' + str(torch.version.cuda))",
                    "cap = torch.cuda.get_device_capability()",
                    "print('capability=' + '.'.join(map(str, cap)))",
                ]
            )
        result = subprocess.run(
            [str(python), "-c", ";".join(smoke_code)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            details = (result.stderr or result.stdout).strip()
            raise SystemExit(f"Smoke test failed for {entry['name']}: {details}")
        summary = ", ".join(line for line in result.stdout.splitlines() if line)
        print(
            f"{entry['name']}: PASS ({target}, Python {entry['python']}"
            + (f"; {summary}" if summary else "")
            + ")"
        )
    print(f"Verified {len(entries)} installed venv environments: PASS")


def verify_submodules(manifest: dict) -> None:
    project_root = HERE.parent
    expected_modules = manifest.get("submodules", {})
    if not expected_modules:
        raise SystemExit("Release manifest does not define any submodule pins.")
    failures = []
    for relative, expected_commit in sorted(expected_modules.items()):
        path = (project_root / relative).resolve()
        if project_root not in path.parents or not path.is_dir():
            failures.append((relative, expected_commit, "missing"))
            continue
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        actual = result.stdout.strip() if result.returncode == 0 else "invalid"
        if actual != expected_commit:
            failures.append((relative, expected_commit, actual))
            continue
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        suffix = " (local tracked modifications present)" if dirty else ""
        print(f"{relative}: PASS ({actual}{suffix})")
    if failures:
        for relative, expected, actual in failures:
            print(f"{relative}: expected {expected}, found {actual}")
        raise SystemExit(
            "Submodule verification failed. Sync the pinned source tree from "
            "the Git/resource relay; offline setup will not contact GitHub."
        )
    print(f"Verified {len(expected_modules)} pinned submodules: PASS")


def main() -> None:
    args = parse_args()
    manifest, _ = load_release(args.release)
    release_directory = resolve_release_directory(args, manifest)
    if args.command == "sync-assets":
        sync_assets(manifest, release_directory)
    elif args.command == "verify-assets":
        verify_assets(manifest, release_directory)
    elif args.command == "prepare-runtime":
        prepare_runtime(release_directory)
    elif args.command == "verify-runtime":
        verify_runtime(release_directory)
    elif args.command == "pack-conda":
        pack_conda_environments(
            manifest, release_directory, args.environment, args.force
        )
    elif args.command == "verify-conda-packs":
        verify_conda_packs(manifest, release_directory, args.environment)
    elif args.command == "install-conda":
        install_conda_environments(manifest, release_directory, args.environment)
    elif args.command == "verify-installed-conda":
        verify_installed_conda(manifest, release_directory, args.environment)
    elif args.command == "sync-uv-cache":
        sync_uv_cache(manifest, release_directory)
    elif args.command == "verify-uv-cache":
        verify_uv_cache(manifest, release_directory)
    elif args.command == "install-venvs":
        install_venv_environments(
            manifest, release_directory, args.environment
        )
    elif args.command == "verify-installed-venvs":
        verify_installed_venvs(manifest, release_directory, args.environment)
    else:
        verify_submodules(manifest)


if __name__ == "__main__":
    main()
