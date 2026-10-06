#!/bin/bash
# Worklog 73: fetch the BEDLAM v1 6fps image tars, the official processed
# training labels (all_npz_12_training) and the SMPL-X v1.1 body model from
# the official MPI-IS download service.  Run BY THE OWNER on LabServer63; the
# script reads ~/.animcv_credentials/{bedlam,smplx} (username=/password= lines)
# and never prints them.  Restart-safe (wget --continue); each file's size and
# sha256 are appended to $ROOT/download_manifest.tsv.
#
# Scene list = the 30 BEDLAM v1 scene folders of the official BEDLAM repo
# (pixelite1201/BEDLAM@f1828fd data_processing/bedlam_scene_names.csv; the
# hair scenes exist only as 30fps tars).  AGORA images are NOT fetched (one
# external source only).
set -uo pipefail
ROOT="${ROOT:-$HOME/animcv-data/bedlam}"
CRED="$HOME/.animcv_credentials"
mkdir -p "$ROOT/images" "$ROOT/labels" "$ROOT/body_models"
LOG="$ROOT/download.log"

field() { sed -n "s/^$2=//p" "$CRED/$1" | head -n1; }
urle() { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~_-] ]] && printf '%s' "${x}" || printf '%%%02X' "'${x}"; done; }
post() { printf 'username=%s&password=%s' "$(urle "$(field "$1" username)")" "$(urle "$(field "$1" password)")"; }

fetch() {  # domain credential sfile dest
  local domain=$1 cred=$2 sfile=$3 dest=$4
  if [ -s "$dest" ] && grep -q "	$dest$" "$ROOT/download_manifest.tsv" 2>/dev/null; then
    echo "SKIP $dest" | tee -a "$LOG"; return 0; fi
  echo "GET $domain:$sfile -> $dest" | tee -a "$LOG"
  local ok=0
  for attempt in 1 2 3; do
    if wget -q --post-data "$(post "$cred")" \
        "https://download.is.tue.mpg.de/download.php?domain=${domain}&resume=1&sfile=${sfile}" \
        -O "$dest" --continue; then ok=1; break; fi
    echo "  retry $attempt failed" | tee -a "$LOG"; sleep 5
  done
  # The service answers an unknown file / bad login with a small HTML page.
  if [ $ok -eq 1 ] && file "$dest" | grep -qi html; then ok=0; mv "$dest" "$dest.html"; fi
  if [ $ok -ne 1 ]; then echo "FAILED $domain:$sfile" | tee -a "$LOG"; return 1; fi
  printf '%s\t%s\t%s\t%s\n' "$(stat -c%s "$dest")" "$(sha256sum "$dest" | cut -d' ' -f1)" "$domain:$sfile" "$dest" \
    >> "$ROOT/download_manifest.tsv"
  echo "  OK $(stat -c%s "$dest") bytes" | tee -a "$LOG"
}

fetch smplx smplx models_smplx_v1_1.zip "$ROOT/body_models/models_smplx_v1_1.zip"
fetch bedlam bedlam bedlam_labels/all_npz_12_training.zip "$ROOT/labels/all_npz_12_training.zip"

SCENES_6FPS="20221010_3_1000_batch01hand 20221010_3-10_500_batch01hand_zoom_suburb_d
20221011_1_250_batch01hand_closeup_suburb_a 20221011_1_250_batch01hand_closeup_suburb_b
20221011_1_250_batch01hand_closeup_suburb_c 20221011_1_250_batch01hand_closeup_suburb_d
20221012_1_500_batch01hand_closeup_highSchoolGym 20221012_3-10_500_batch01hand_zoom_highSchoolGym
20221013_3-10_500_batch01hand_static_highSchoolGym 20221013_3_250_batch01hand_static_bigOffice
20221013_3_250_batch01hand_orbit_bigOffice 20221014_3_250_batch01hand_orbit_archVizUI3_time15
20221015_3_250_batch01hand_orbit_archVizUI3_time19 20221015_3_250_batch01hand_orbit_archVizUI3_time12
20221015_3_250_batch01hand_orbit_archVizUI3_time10 20221017_3_1000_batch01hand
20221018_1_250_batch01hand_zoom_suburb_b 20221018_3-8_250_batch01hand
20221018_3_250_batch01hand_orbit_archVizUI3_time15 20221018_3-8_250_batch01hand_pitchUp52_stadium
20221018_3-8_250_batch01hand_pitchDown52_stadium 20221019_3_250_highbmihand
20221019_1_250_highbmihand_closeup_suburb_b 20221019_1_250_highbmihand_closeup_suburb_c
20221019_3-8_250_highbmihand_orbit_stadium 20221019_3-8_1000_highbmihand_static_suburb_d
20221020-3-8_250_highbmihand_zoom_highSchoolGym_a"
SCENES_30FPS="20221022_3_250_batch01handhair_static_bigOffice 20221024_10_100_batch01handhair_zoom_suburb_d
20221024_3-10_100_batch01handhair_static_highSchoolGym"

for s in $SCENES_6FPS; do fetch bedlam bedlam "bedlam_images_train/${s}_6fps.tar" "$ROOT/images/${s}_6fps.tar"; done
for s in $SCENES_30FPS; do fetch bedlam bedlam "bedlam_images_train/${s}_30fps.tar" "$ROOT/images/${s}_30fps.tar"; done
echo "DONE $(grep -c FAILED "$LOG") failures" | tee -a "$LOG"
