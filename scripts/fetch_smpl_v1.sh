#!/usr/bin/env bash
# Worklog 97: download the official SMPLify package (contains the SMPL v1 neutral body model required by HMR 2.0)
# with YOUR OWN smplify.is.tue.mpg.de account. Run it interactively on LabServer63, e.g. from the Mac:
#     ssh -t LabServer63 'bash ~/AnimCV/scripts/fetch_smpl_v1.sh'            (default file name)
#     ssh -t LabServer63 'bash ~/AnimCV/scripts/fetch_smpl_v1.sh <sfile>'    (file name from the Download link)
# Credentials are read with a hidden prompt, passed to curl through mode-600 temp files (never on the command line,
# so they do not appear in `ps`), and deleted on exit. Nothing is stored. Only the neutral SMPL v1 .pkl is extracted.
set -euo pipefail

SFILE="${1:-mpips_smplify_public_v2.zip}"  # the file name shown in the site's Download link (sfile=...)
DEST="${2:-$HOME/animcv-data/body_models/smpl_v1}"
URL="https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=${SFILE}"
PKL='basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
ZIP="$DEST/${SFILE##*/}"

mkdir -p "$DEST"
umask 077
UF=$(mktemp) PF=$(mktemp)
trap 'rm -f "$UF" "$PF" "$ZIP.part"' EXIT

IFS= read -rp "smplify.is.tue.mpg.de username (e-mail): " U
IFS= read -rsp "password (hidden): " P
echo
U="${U//[[:space:]]/}"  # e-mail logins never contain whitespace (a pasted tab/space breaks the login); password is kept verbatim
echo "using username: '$U'"
printf '%s' "$U" > "$UF"
printf '%s' "$P" > "$PF"
unset U P

echo "Downloading to $ZIP ..."
HTTP=$(curl -L --retry 3 --progress-bar -o "$ZIP.part" -w "%{http_code}" \
  --data-urlencode "username@$UF" --data-urlencode "password@$PF" "$URL")
rm -f "$UF" "$PF"
if [ "$HTTP" = "403" ]; then
  echo "ERROR 403: login accepted but access to '$SFILE' was refused (licence for this download not accepted on the site's Download page, or a different file name)." >&2
  exit 1
fi
if [ "$HTTP" = "401" ]; then
  echo "ERROR 401: smplify.is.tue.mpg.de rejected the login (check e-mail activation, licence acceptance, and that the username is the registered e-mail)." >&2
  exit 1
fi

# A failed login returns an HTML page with HTTP 200; a real archive starts with the ZIP magic "PK".
if [ "$(head -c 2 "$ZIP.part")" != "PK" ]; then
  cp "$ZIP.part" "$DEST/last_response.txt"
  grep -qi "Sign in" "$ZIP.part" && echo "ERROR: the login was not accepted (the site returned its sign-in page)." >&2
  echo "ERROR: HTTP $HTTP, the server did not return a ZIP archive ($(wc -c < "$ZIP.part") bytes)." >&2
  echo "Response kept at $DEST/last_response.txt (no credentials in it). First lines, tags stripped:" >&2
  sed -e 's/<[^>]*>/ /g' "$ZIP.part" | tr -s ' \t' ' ' | grep -v '^ *$' | head -25 >&2
  exit 1
fi
mv "$ZIP.part" "$ZIP"

python3 - "$ZIP" "$DEST" "$PKL" <<'EOF'
import sys, zipfile, os, hashlib
zpath, dest, name = sys.argv[1:]
with zipfile.ZipFile(zpath) as z:
    hits = [n for n in z.namelist() if n.endswith("/" + name) or n == name]
    if not hits:
        print("ERROR: %s not found in the archive. .pkl entries:" % name, file=sys.stderr)
        print("\n".join(n for n in z.namelist() if n.endswith(".pkl")), file=sys.stderr)
        sys.exit(1)
    out = os.path.join(dest, name)
    with z.open(hits[0]) as src, open(out, "wb") as dst:
        dst.write(src.read())
print("extracted:", out)
print("sha256:", hashlib.sha256(open(out, "rb").read()).hexdigest())
EOF
echo "Done. Tell Claude: SMPL v1 is at $DEST/$PKL"
