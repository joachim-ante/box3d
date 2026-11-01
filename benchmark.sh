#!/usr/bin/env bash

# Build the Box3D benchmark app in Release and run a single benchmark.
# Usage: ./benchmark.sh <name|index> [extra benchmark args]
#   ./benchmark.sh ray_cast
#   ./benchmark.sh ray_cast -w=1 -r=2
# Run ./benchmark.sh -l to list the registered benchmarks.
#
# Uses build-release/ so the Debug build in build/ is left alone.

set -u

cd "$(dirname "$0")"

BUILD_DIR=build-release
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

if [ $# -eq 0 ]; then
	echo "Usage: $0 <name|index> [extra benchmark args]"
	exit 1
fi

if ! cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBOX3D_BENCHMARKS=ON -DBOX3D_SAMPLES=OFF \
	-DBOX3D_UNIT_TESTS=OFF -DBUILD_SHARED_LIBS=OFF > "$LOG" 2>&1; then
	echo "CMake configure failed:"
	cat "$LOG"
	exit 1
fi

if ! cmake --build "$BUILD_DIR" --target benchmark --parallel > "$LOG" 2>&1; then
	echo "Build failed:"
	grep -E "error|Error" "$LOG" || tail -n 30 "$LOG"
	exit 1
fi

# Single-config generators put the binary in bin/, multi-config (Xcode, VS) in bin/<Config>/
BENCH_BIN=""
for candidate in "$BUILD_DIR/bin/benchmark" "$BUILD_DIR/bin/Release/benchmark" "$BUILD_DIR/bin/Release/benchmark.exe"; do
	if [ -x "$candidate" ]; then
		BENCH_BIN=$candidate
		break
	fi
done

if [ -z "$BENCH_BIN" ]; then
	echo "Benchmark binary not found under $BUILD_DIR/bin"
	exit 1
fi

if [ "$1" = "-l" ] || [ "$1" = "--list" ]; then
	exec "$BENCH_BIN" -l
fi

NAME=$1
shift
exec "$BENCH_BIN" -b="$NAME" "$@"
