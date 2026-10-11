#!/bin/bash
# Worklog 73: fetch an explicit list of BEDLAM files (one "<sfile> <dest-relative-path>"
# per line in $1) from the official MPI-IS download service.  Run BY THE OWNER
# on LabServer63 inside tmux; the BEDLAM login is asked once and passed to wget
# through a private temp file (never on a command line).  A reply smaller than
# 1 kB, or a text/HTML reply, is a failure.  Restart-safe; size and sha256 are
# appended to $ROOT/download_manifest_files.tsv.
set -uo pipefail
LIST=$1
ROOT="${ROOT:-$HOME/animcv-data/bedlam}"
LOG="$ROOT/download_files.log"
MANIFEST="$ROOT/download_manifest_files.tsv"

read -r -p "BEDLAM website email/username: " u; read -r -s -p "BEDLAM website password: " pw; echo
urle() { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~_-] ]] && printf '%s' "${x}" || printf '%%%02X' "'${x}"; done; }
POSTDIR=$(umask 077; mktemp -d)
trap 'rm -rf "$POSTDIR"' EXIT
(umask 077; printf 'username=%s&password=%s' "$(urle "$u")" "$(urle "$pw")" > "$POSTDIR/bedlam")
unset u pw

while read -r sfile rel; do
  [ -z "$sfile" ] && continue
  dest="$ROOT/$rel"; mkdir -p "$(dirname "$dest")"
  if grep -q "	$dest$" "$MANIFEST" 2>/dev/null; then echo "SKIP $dest" | tee -a "$LOG"; continue; fi
  echo "GET $sfile" | tee -a "$LOG"
  ok=0
  for attempt in 1 2 3; do
    if wget -q --post-file "$POSTDIR/bedlam" \
        "https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=${sfile}" \
        -O "$dest" --continue; then ok=1; break; fi
    echo "  retry $attempt failed" | tee -a "$LOG"; sleep 5
  done
  if [ $ok -eq 1 ] && { [ "$(stat -c%s "$dest")" -lt 1000 ] || file "$dest" | grep -qi -e text -e html; }; then
    echo "  bad reply: $(head -c 80 "$dest" | tr -d '\n')" | tee -a "$LOG"; mv "$dest" "$dest.bad"; ok=0
  fi
  if [ $ok -ne 1 ]; then echo "FAILED $sfile" | tee -a "$LOG"; continue; fi
  printf '%s\t%s\t%s\t%s\n' "$(stat -c%s "$dest")" "$(sha256sum "$dest" | cut -d' ' -f1)" "bedlam:$sfile" "$dest" >> "$MANIFEST"
  echo "  OK $(stat -c%s "$dest") bytes" | tee -a "$LOG"
done < "$LIST"
echo "DONE $(grep -c '^FAILED' "$LOG") failures" | tee -a "$LOG"
