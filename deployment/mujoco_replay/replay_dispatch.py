#!/usr/bin/env python3
"""Dispatch a Retargeting replay to hand-only or full-arm presentation.

``viewpoint`` describes the input. ``render-mode`` describes the output. In
automatic mode an ego capture gets the free-root Sharpa hand/object viewer,
while an exo capture gets the UR3e IK viewer. Unknown legacy captures keep the
historical full-arm behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FULL_ARM_ENTRYPOINT = HERE / "replay_retarget.py"
HAND_ONLY_ENTRYPOINT = REPO_ROOT / "retargeting" / "replay_viser.py"
DEFAULT_TRAJ = Path("trajectory_mjwp.npz")


def _capture_viewpoint(trajectory: Path) -> str:
    sidecar = trajectory.expanduser().resolve().parent / "capture_metadata.json"
    if not sidecar.is_file():
        return "auto"
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "auto"
    viewpoint = payload.get("viewpoint", "auto")
    return viewpoint if viewpoint in {"auto", "ego", "exo"} else "auto"


def resolve_render_mode(render_mode: str, viewpoint: str) -> str:
    """Resolve ``auto`` without coupling explicit user choices."""
    if render_mode != "auto":
        return render_mode
    if viewpoint == "ego":
        return "hand-only"
    # Preserve the historical behavior for exo and legacy/unknown captures.
    return "full-arm"


def _parse_dispatch_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--render-mode", choices=("auto", "hand-only", "full-arm"), default="auto")
    parser.add_argument("--viewpoint", choices=("auto", "ego", "exo"), default=None)
    parser.add_argument("--camera-motion", choices=("auto", "moving", "static"), default=None)
    parser.add_argument(
        "--camera-mode",
        choices=("auto", "ego", "scene", "top-down"),
        default="auto",
    )
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--ego-fov", type=float, default=None)
    parser.add_argument("--ego-distance", type=float, default=None)
    parser.add_argument(
        "--display-object-upright",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--display-object-up-axis",
        choices=("+x", "-x", "+y", "-y", "+z", "-z"),
        default=None,
    )
    parser.add_argument(
        "--display-object-up-vector",
        type=float,
        nargs=3,
        default=None,
    )
    parser.add_argument(
        "--display-hand-level",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_known_args(argv)


def main() -> None:
    args, remaining = _parse_dispatch_args(sys.argv[1:])
    stored_viewpoint = _capture_viewpoint(args.traj)
    viewpoint = args.viewpoint or stored_viewpoint
    resolved = resolve_render_mode(args.render_mode, viewpoint)
    print(
        "Replay dispatch: "
        f"viewpoint={viewpoint}, requested={args.render_mode}, resolved={resolved}"
    )

    if resolved == "full-arm":
        # The full-arm parser owns all existing options. It also accepts the
        # presentation metadata so final packages retain the user's override.
        command = [sys.executable, str(FULL_ARM_ENTRYPOINT), *sys.argv[1:]]
        os.execv(sys.executable, command)

    if "--help" in remaining or "-h" in remaining:
        os.execv(
            sys.executable,
            [sys.executable, str(HAND_ONLY_ENTRYPOINT), "--help"],
        )
    if remaining:
        print(
            "Hand-only mode ignores full-arm-only arguments: "
            + " ".join(remaining)
        )
    fps = max(1.0, 60.0 * max(args.speed, 0.01))
    command = [
        sys.executable,
        str(HAND_ONLY_ENTRYPOINT),
        "--run-dir",
        str(args.traj.expanduser().resolve().parent),
        "--traj",
        str(args.traj.expanduser().resolve()),
        "--viewpoint",
        viewpoint,
        "--camera-mode",
        (
            args.camera_mode
            if args.camera_mode != "auto"
            else ("ego" if viewpoint == "ego" else "scene")
        ),
        "--fps",
        str(fps),
    ]
    if args.port is not None:
        command.extend(("--port", str(args.port)))
    if args.camera_motion is not None:
        command.extend(("--camera-motion", args.camera_motion))
    if args.ego_fov is not None:
        command.extend(("--ego-fov", str(args.ego_fov)))
    if args.ego_distance is not None:
        command.extend(("--ego-distance", str(args.ego_distance)))
    if args.display_object_upright is not None:
        command.append(
            "--display-object-upright"
            if args.display_object_upright
            else "--no-display-object-upright"
        )
    if args.display_object_up_axis is not None:
        command.append(f"--display-object-up-axis={args.display_object_up_axis}")
    if args.display_object_up_vector is not None:
        command.append("--display-object-up-vector")
        command.extend(str(value) for value in args.display_object_up_vector)
    if args.display_hand_level is not None:
        command.append(
            "--display-hand-level"
            if args.display_hand_level
            else "--no-display-hand-level"
        )
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
