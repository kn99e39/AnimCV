#!/bin/bash
# Worklog 73: fetch the 30fps PNG archives of the four scenes the official
# BEDLAM repo uses for validation (pixelite1201/BEDLAM@f1828fd
# train/core/config.py DATASET_FILES[0]).  Their images are not distributed as
# 6fps tars (the service answers "Error: File not found."), only inside the
# b0 30fps PNG release, five archives per scene (official be_download.sh
# naming: b0/<scene>/png/<scene>_png.<0..4>.tar).  Labels for these scenes are
# already in all_npz_12_training.
#
# Run BY THE OWNER on LabServer63 inside tmux; the BEDLAM login is asked once
# and passed to wget via a private temp file (never on a command line).  A
# reply smaller than 1 MB or a text/HTML reply is treated as a failure.
# Restart-safe (wget --continue); size and sha256 go to
# $ROOT/download_manifest_validation.tsv.
set -uo pipefail
ROOT="${ROOT:-$HOME/animcv-data/bedlam}"
OUT="$ROOT/validation_png"
LOG="$ROOT/download_validation.log"
MANIFEST="$ROOT/download_manifest_validation.tsv"
mkdir -p "$OUT"

read -r -p "BEDLAM website email/username: " u; read -r -s -p "BEDLAM website password: " pw; echo
urle() { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~_-] ]] && printf '%s' "${x}" || printf '%%%02X' "'${x}"; done; }
POSTDIR=$(umask 077; mktemp -d)
trap 'rm -rf "$POSTDIR"' EXIT
(umask 077; printf 'username=%s&password=%s' "$(urle "$u")" "$(urle "$pw")" > "$POSTDIR/bedlam")
unset u pw

SCENES="20221019_3-8_250_highbmihand_orbit_stadium 20221018_3_250_batch01hand_orbit_archVizUI3_time15
20221018_1_250_batch01hand_zoom_suburb_b 20221018_3-8_250_batch01hand"

for s in $SCENES; do
  for part in 0 1 2 3 4; do
    name="${s}_png.${part}.tar"; dest="$OUT/$name"; sfile="b0/${s}/png/${name}"
    if grep -q "	$dest$" "$MANIFEST" 2>/dev/null; then echo "SKIP $dest" | tee -a "$LOG"; continue; fi
    echo "GET $sfile" | tee -a "$LOG"
    ok=0
    for attempt in 1 2 3; do
      if wget -q --post-file "$POSTDIR/bedlam" \
          "https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=${sfile}" \
          -O "$dest" --continue; then ok=1; break; fi
      echo "  retry $attempt failed" | tee -a "$LOG"; sleep 5
    done
    if [ $ok -eq 1 ] && { [ "$(stat -c%s "$dest")" -lt 1000000 ] || file "$dest" | grep -qi -e text -e html; }; then
      echo "  bad reply: $(head -c 80 "$dest" | tr -d '\n')" | tee -a "$LOG"; mv "$dest" "$dest.bad"; ok=0
    fi
    if [ $ok -ne 1 ]; then echo "FAILED $sfile" | tee -a "$LOG"; continue; fi
    printf '%s\t%s\t%s\t%s\n' "$(stat -c%s "$dest")" "$(sha256sum "$dest" | cut -d' ' -f1)" "bedlam:$sfile" "$dest" >> "$MANIFEST"
    echo "  OK $(stat -c%s "$dest") bytes" | tee -a "$LOG"
  done
done
echo "DONE $(grep -c '^FAILED' "$LOG") failures" | tee -a "$LOG"
