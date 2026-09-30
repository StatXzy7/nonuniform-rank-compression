#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/target/verify"

if [[ -n "${JAVA_HOME:-}" ]]; then
  JAVAC="$JAVA_HOME/bin/javac"
  JAVA="$JAVA_HOME/bin/java"
else
  JAVAC="javac"
  JAVA="java"
fi

echo "[1/3] Compiling Java sources and tests..."
mkdir -p "$BUILD_DIR"
find "$ROOT_DIR/src/main/java" "$ROOT_DIR/src/test/java" -name '*.java' -print0 \
  | xargs -0 "$JAVAC" -encoding UTF-8 -Xlint:all -Werror -d "$BUILD_DIR"

echo "[2/3] Running automated checks..."
"$JAVA" -cp "$BUILD_DIR" lowrank.TestEngine

echo "[3/3] Running finite small example..."
"$JAVA" -cp "$BUILD_DIR" Example

echo "Verification completed successfully."
