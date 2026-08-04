#!/bin/bash
# Reassembles datasets/mpi_inf_3dhp/**/annot.mat from the committed
# annot.mat.part-* chunks (each <=90MB, split to stay under GitHub's per-file
# limit and avoid Git LFS) and verifies each result against the SHA-256
# recorded in datasets/mpi_inf_3dhp/intake_manifest.json.
#
# Safe to re-run: skips any annot.mat that already exists and matches its
# recorded checksum.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="$REPO_ROOT/datasets/mpi_inf_3dhp"
MANIFEST="$DATA_ROOT/intake_manifest.json"

expected_sha256() {
  # $1 = "S<n>/Seq<m>" suffix
  python3 - "$MANIFEST" "$1" <<'PY'
import json, sys
manifest, suffix = sys.argv[1], sys.argv[2]
data = json.load(open(manifest))
for entry in data["sequences"]:
    if entry["annotation_path"].endswith(f"{suffix}/annot.mat"):
        print(entry["annotation_sha256"])
        break
else:
    sys.exit(f"no manifest entry for {suffix}")
PY
}

for parts in "$DATA_ROOT"/S*/Seq*/annot.mat.part-00; do
  dir="$(dirname "$parts")"
  out="$dir/annot.mat"
  suffix="$(basename "$(dirname "$dir")")/$(basename "$dir")"
  want="$(expected_sha256 "$suffix")"
  if [ -f "$out" ]; then
    got="$(sha256sum "$out" | cut -d' ' -f1)"
    if [ "$got" = "$want" ]; then
      echo "OK (already present): $suffix/annot.mat"
      continue
    fi
    echo "MISMATCH, rebuilding: $suffix/annot.mat"
    rm -f "$out"
  fi
  cat "$dir"/annot.mat.part-* > "$out"
  got="$(sha256sum "$out" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "CHECKSUM MISMATCH after reassembly: $suffix/annot.mat (got $got, want $want)" >&2
    exit 1
  fi
  echo "reassembled: $suffix/annot.mat"
done
