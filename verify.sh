#!/usr/bin/env bash
# Compiles the engine and runs the automated test suite plus the small demo.
# No external services are contacted; the only third-party dependency is EJML,
# expected under lib/ (see lib/fetch_jars.sh for offline preparation).
set -euo pipefail
cd "$(dirname "$0")"

CP="lib/*"

shopt -s nullglob
jars=(lib/*.jar)
if [ "${#jars[@]}" -eq 0 ]; then
  echo "ERROR: no jars found in lib/. Run lib/fetch_jars.sh first (online once)," >&2
  echo "       or copy ejml-core/ejml-ddense/ejml-simple 0.43.1 jars there offline." >&2
  exit 2
fi

echo ">> Cleaning build output"
mkdir -p build
find build -type f -name '*.class' -delete

echo ">> Compiling main sources"
javac -cp "$CP" -d build $(find src -name '*.java')

echo ">> Compiling tests"
javac -cp "$CP:build" -d build $(find test -name '*.java')

echo ">> Running automated tests"
java -cp "$CP:build" com.lowrank.EngineTest

echo
echo ">> Running small end-to-end demo"
java -cp "$CP:build" com.lowrank.Demo >/dev/null

echo
echo "VERIFY OK"
