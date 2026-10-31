# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Box3D is a 3D rigid body physics engine for games by Erin Catto, written in portable C17 (samples are C++20). The core library (`src/`, public headers in `include/box3d/`) has no dependencies beyond the C runtime. Public types and functions use the `b3` prefix.

## Build and test

CMake 3.22+. Presets live in `CMakePresets.json`; the build directory is always `build/`.

```sh
# macOS (Xcode generator, multi-config)
cmake --preset macos
cmake --build --preset macos-debug          # or macos-release
./build/bin/Debug/test                      # unit tests
./build/bin/Debug/samples                   # samples app, run from repo root

# Linux
cmake --preset linux-debug && cmake --build --preset linux-debug
./build/bin/test

# Windows (MSVC)
cmake --preset windows && cmake --build --preset windows-debug
.\build\bin\Debug\test.exe
```

The CI configure line is the one to match when checking a change compiles cleanly:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBOX3D_SAMPLES=OFF -DBUILD_SHARED_LIBS=OFF -DBOX3D_VALIDATE=ON -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
cmake --build build --parallel
```

### Running tests

Quickest path: `./test.sh` builds the `test` target in `build/` and runs it, printing `All tests succeeded` or the failing suite and condition. `./test.sh JointTest` runs one suite.

The unit test binary is a single executable with no framework. `test/main.c` lists every suite and runs them in order; it stops at the first failure.

- Run everything: `./build/bin/Debug/test`
- Run one suite: pass its function name as the only argument, e.g. `./build/bin/Debug/test JointTest`. Names are the `XxxTest` functions declared at the top of `test/main.c` (e.g. `DeterminismTest`, `ManifoldTest`, `SeparatingAxisTest`).
- Adding a suite: create `test/test_foo.c` with `int FooTest(void)`, add it to `test/CMakeLists.txt`, declare it and add a `MAYBE_RUN_TEST( FooTest )` line in `test/main.c`. Use `ENSURE`, `ENSURE_SMALL`, and `RUN_SUBTEST` from `test/test_macros.h`; a test returns 1 on failure.
- Tests link the static library and include `src/` directly, so they may call internal functions. Tests are skipped when `BUILD_SHARED_LIBS=ON`.

### Other targets and options

- `samples`: built by default. Headless run for checking a sample: `./build/bin/Debug/samples --frames N --sample N` (sample index is the sorted registry index). `--replay <file.b3rec>` opens a recording, `--zup` switches up axis.
- `benchmark`: `-DBOX3D_BENCHMARKS=ON`. `./build/bin/Debug/benchmark -l` lists benchmarks, `-b <name>` runs one, `-sat` compares GJK vs SAT, `-simd 4|8` forces SIMD width. Reference results per CPU live in `benchmark/`.
- `doc`: `-DBOX3D_DOCS=ON` (needs Doxygen). The manual is `docs/*.md`.
- `BOX3D_SANITIZE=ON` with `BOX3D_SANITIZER_TYPE=address|thread|undefined|memory`. CI runs ASan+UBSan on macOS/Windows, TSan and MSan on Linux clang.
- `BOX3D_DOUBLE_PRECISION=ON`, `BOX3D_DISABLE_SIMD=ON`, `BOX3D_PROFILE=ON` (Tracy), `BOX3D_VALIDATE` (heavy validation, debug builds only, on by default).
- FetchContent downloads (imgui, implot, nfd, Tracy) are cached in `.fetchcontent-cache/` at the repo root so `rm -rf build` does not re-download.

### Style

Format with `clang-format` using the repo `.clang-format` (Microsoft base, tabs, 100 columns, `PointerAlignment: Left`, spaces inside parens: `if ( x )`, `foo( a, b )`). Allman braces. No compiler warnings; the library builds with `-Wall -Wextra -pedantic -Wmissing-prototypes` or MSVC `/Wall`. Internal (non-static) functions must have a prototype in a header. Public API needs Doxygen comments. Every source file starts with the SPDX copyright and MIT license lines.

## Architecture

### Public API vs internals

`include/box3d/` is the entire public surface. Users hold opaque ids (`b3BodyId`, `b3ShapeId`, `b3JointId`, ...) defined in `id.h`; each id packs a 1-based index, a world index, and a generation. All internal state is reached through `b3World` (`src/physics_world.h`), looked up from the id with generation checks. `base.h` defines `B3_ASSERT` (active without `NDEBUG` or with `B3_ENABLE_ASSERT`, which RelWithDebInfo sets) and `B3_VALIDATE` (only with `BOX3D_VALIDATE` and no `NDEBUG`).

### World data layout: solver sets

The central data-oriented idea (`src/solver_set.h`). Bodies are split into three structs:

- `b3Body` (`body.h`): bookkeeping not used by the solver, stable id, lives in `world->bodies`.
- `b3BodySim`: transform, mass, forces; lives in a solver set's `bodySims` array.
- `b3BodyState`: velocity and delta position/rotation, 64 bytes; exists only for the awake set.

Solver sets (`world->solverSets`) are: `b3_staticSet`, `b3_disabledSet`, `b3_awakeSet`, then one set per sleeping island. Putting an island to sleep moves its bodies, contacts, and joints into a new set; waking merges it back into the awake set. Pointers into set arrays are invalidated by sleep/wake/merge, so code re-looks-up by id or `setIndex` + `localIndex`. Joints follow the same `b3Joint`/`b3JointSim` split. Contacts are a single `b3Contact` (`contact.h`) referenced from sets by index; the per-color constraint data lives in `constraint_graph.c`.

### Step pipeline

`b3World_Step` in `physics_world.c` runs: update broad-phase pairs (`broad_phase.c`, built on `dynamic_tree.c`) -> `b3Collide` (narrow phase over awake contacts, parallel, produces manifolds and contact begin/end state bits) -> `b3Solve` in `solver.c`. The solver runs the *Soft Step* substepping scheme as a sequence of `b3_stage*` stages (prepare joints/contacts, integrate velocities, warm start, solve, integrate positions, relax, restitution, store impulses) over the graph-colored constraints in `constraint_graph.c`. Each stage is distributed to worker threads through `b3ParallelFor` / the task callbacks in `b3WorldDef` (`parallel_for.c`, `scheduler.c`). Per-worker scratch state is `b3TaskContext`. Continuous collision, island splitting, and sleep candidates are handled after the solve. Island bookkeeping for sleep is in `island.c`.

### SIMD and wide code

`src/core.h` selects the SIMD path from compile definitions: SSE2/AVX2 on x86, NEON on AArch64, SSE2 mapping on wasm, or scalar with `BOX3D_DISABLE_SIMD`. The wide contact solver and convex manifold code are written once as `.inl` files (`contact_solver_wide.inl`, `convex_manifold_wide.inl`) and instantiated twice: `wide4.c` defines `B3_SIMD_WIDTH 4` and `wide8.c` defines width 8 and compiles with AVX2 target attributes (`/arch:AVX2` on MSVC). AVX2 availability is detected at runtime. Any code that includes `src/` headers (tests, `shared/sat_benchmark.c`) must receive the same `BOX3D_SIMD_DEFINITIONS`/`BOX3D_SIMD_OPTIONS`, which the top-level CMake propagates.

### Determinism

Cross-platform determinism is a hard requirement. CMake adds `-ffp-contract=off` on gcc/clang; do not introduce FMA, platform-dependent math, or order-dependent parallel reductions. `test/test_determinism.c` checks golden hashes (`RAGDOLL_HASH`, `WAVE_PILE_HASH`, etc., with separate values for double precision) across worker counts and across platforms. A legitimate simulation change that alters results requires updating those hashes; CI runs them on gcc, clang, MSVC, MinGW, ARM64 and at -O2 (RelWithDebInfo), so a hash that only matches on one platform indicates a real determinism bug. Scenarios live in `shared/determinism.c` so samples can display the same scenes.

### Collision

Shapes: sphere, capsule, convex hull (`hull.c`), triangle mesh (`mesh.c`, `mesh_contact.c`), height field, and compound. Narrow phase produces `b3Manifold` with up to `B3_MAX_MANIFOLD_POINTS` points (`manifold.c`, `convex_manifold.c`, `triangle_manifold.c`); GJK is in `distance.c`; the hull-vs-hull SAT with its `b3SATCache` is in `convex_manifold.c`. Manifolds are allocated from per-point-count block allocators on the world. Sensors (`sensor.c`) are processed separately from contacts. The character mover (`mover.c`) is a kinematic collide-and-slide helper on top of the query API.

### Recording and replay

`recording.c` / `replay.c` serialize world creation ops and steps into `.b3rec` files (header `B3RC`, versioned by `B3_REC_VERSION_MAJOR/MINOR` in `recording.h`). Bump the major version when a writer changes layout. `world_snapshot.c` hashes world state for determinism checks.

### Supporting layers

- `shared/`: C code used by tests, samples, and benchmarks (benchmark scenes, determinism scenes, ragdoll `human.c`). Link target `shared`.
- `samples/`: C++20 app on sokol (D3D11/Metal/OpenGL) + imgui + implot + ImGuizmo. Each sample is a `Sample` subclass registered with a file-scope `RegisterSample( "Category", "Name", Create )` call; adding a sample is adding a class and that line in the relevant `sample_*.cpp`. Renderer is in `samples/gfx/`, platform host in `samples/host/`. Shaders are prebuilt headers; set `BOX3D_BUILD_SHADERS=ON` to recompile with sokol-shdc.
- `data/`: meshes, hulls, and tree assets loaded by samples; the samples app must run with the repo root as working directory.

## Contributing rules (from CONTRIBUTING.md and README.md)

- Pull requests need an issue labeled `ready-for-pr` first, and the PR description must contain `Closes #N`. The PR checklist is: run unit tests, run all samples.
- Library code is C17 and must not use C++ features. Samples are C++20.
- Per the README LLM policy, LLM-written code is accepted for unit tests, the samples app, Box2D->Box3D migration, build configuration, code review, and benchmarking. Engine source in `src/` is written by the maintainer.
