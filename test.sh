#!/usr/bin/env bash

# Build Box3D and run the unit tests.
# Prints "All tests succeeded" on success, otherwise lists the failed tests.
# Usage: ./test.sh [TestName]   (optional filter, e.g. ./test.sh JointTest)

set -u

cd "$(dirname "$0")"

BUILD_DIR=build
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

if ! cmake -S . -B "$BUILD_DIR" -DBOX3D_SAMPLES=OFF > "$LOG" 2>&1; then
	echo "CMake configure failed:"
	cat "$LOG"
	exit 1
fi

if ! cmake --build "$BUILD_DIR" --target test --parallel > "$LOG" 2>&1; then
	echo "Build failed:"
	grep -E "error|Error" "$LOG" || tail -n 30 "$LOG"
	exit 1
fi

# Single-config generators put the binary in bin/, multi-config (Xcode, VS) in bin/<Config>/
TEST_BIN=""
for candidate in "$BUILD_DIR/bin/test" "$BUILD_DIR/bin/Debug/test" "$BUILD_DIR/bin/Release/test"; do
	if [ -x "$candidate" ]; then
		TEST_BIN=$candidate
		break
	fi
done

if [ -z "$TEST_BIN" ]; then
	echo "Test binary not found under $BUILD_DIR/bin"
	exit 1
fi

if "$TEST_BIN" "$@" > "$LOG" 2>&1; then
	echo "All tests succeeded"
	exit 0
fi

echo "Tests failed:"
# The runner stops at the first failing suite. Show the failing suite, subtest, and the condition.
grep -E "^(test failed|  subtest failed|condition false)" "$LOG" || tail -n 30 "$LOG"
exit 1
