#!/usr/bin/env bash
# One-time online fetch of the single matrix library (EJML 0.43.1, Apache-2.0).
# Run this once where internet is available; afterwards the project is fully
# offline because verify.sh only uses the jars checked into lib/.
set -euo pipefail
cd "$(dirname "$0")"
VERSION=0.43.1
BASE="https://repo1.maven.org/maven2/org/ejml"
for module in core ddense simple; do
  jar="ejml-${module}-${VERSION}.jar"
  if [ -s "$jar" ]; then
    echo "have $jar"
    continue
  fi
  url="${BASE}/ejml-${module}/${VERSION}/${jar}"
  echo "downloading $url"
  curl -fsSL --retry 3 -o "$jar.tmp" "$url"
  mv "$jar.tmp" "$jar"
done
echo "EJML jars ready in $(pwd)"
