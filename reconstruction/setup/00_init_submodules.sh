#!/bin/bash
# [00] Initialize the third-party submodules at their pinned fork-date commits.
#      (SAM3D, Fast-SAM3D, HaWoR + nested lietorch/eigen, TAPNet.)
# Weights are NOT pulled here — run 02_fetch_weights.sh for those.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export GIT_LFS_SKIP_SMUDGE=1   # don't smudge any LFS blobs; weights come from 02_fetch_weights.sh

echo "[00] Cloning + checking out submodules at pinned commits (this can take a while)..."
for attempt in 1 2 3; do
	if git -c http.version=HTTP/1.1 submodule update --init --recursive; then
		break
	fi
	if [ "$attempt" -eq 3 ]; then
		echo "[00] submodule update failed after $attempt attempts" >&2
		exit 1
	fi
	echo "[00] submodule update failed; retrying ($((attempt + 1))/3)" >&2
done

echo "[00] Submodule pins:"
git submodule status
echo "[00] Done. Next: ./setup/01_create_envs.sh"
