#!/usr/bin/env python3
"""Repair material bindings authored by the Isaac MJCF importer.

The Isaac 5 MJCF importer writes a direct ``DefaultMaterial`` binding on
nearly every mesh.  That direct binding takes precedence over the coloured
material binding on the visual-link Xform, so a correctly coloured MuJoCo
scene renders entirely grey.  Removing only those default bindings lets USD
resolve the intended material from the visual-link parent.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from pxr import Usd, UsdShade


DEFAULT_MATERIAL_PATH = "/World/Looks/DefaultMaterial"


def repair_default_mesh_bindings(stage: Usd.Stage) -> int:
    """Remove leaf-mesh DefaultMaterial bindings and return the repair count."""
    repaired = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        direct = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        if str(direct.GetMaterialPath()) != DEFAULT_MATERIAL_PATH:
            continue
        # Clear the relationship authored directly on the mesh.  The mesh then
        # inherits the binding authored on its visual-link Xform.
        direct.GetBindingRel().ClearTargets(False)
        repaired += 1
    return repaired


def repair_file(input_path: Path, output_path: Path) -> int:
    stage = Usd.Stage.Open(str(input_path))
    if not stage:
        raise RuntimeError(f"Could not open USD stage: {input_path}")
    repaired = repair_default_mesh_bindings(stage)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Export through a sibling temporary file when replacing in place.  This
    # avoids truncating the layer while it is still open in USD.
    if input_path.resolve() == output_path.resolve():
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}.", suffix=output_path.suffix,
            dir=output_path.parent,
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            if not stage.GetRootLayer().Export(str(temporary_path)):
                raise RuntimeError(f"Failed to export repaired USD: {temporary_path}")
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    elif not stage.GetRootLayer().Export(str(output_path)):
        raise RuntimeError(f"Failed to export repaired USD: {output_path}")
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    output = args.output or args.input
    repaired = repair_file(args.input, output)
    print(f"Removed {repaired} direct DefaultMaterial mesh binding(s) from {output}")


if __name__ == "__main__":
    main()
