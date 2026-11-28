#!/usr/bin/env bash
# One command: build, GPU info, fixtures, benchmark CSV, ncu profile per kernel, nsys trace, bundle.
#
#   scripts/run_bench.sh <tag> "<GPU name>" [cuda_arch]
#   scripts/run_bench.sh rtx5080_laptop "RTX 5080 Laptop" 120
#   scripts/run_bench.sh b200 "NVIDIA B200" 100
#
# Outputs:
#   results/<tag>.csv, <tag>_gpu.txt, <tag>_bench.log (ceiling and L2 info)
#   profiles/<tag>/NN_<name>_<len>.ncu-rep   one per kernel
#   profiles/<tag>/nsys_all.nsys-rep
#   <tag>_bundle.tar.gz                      everything above
#
# Optional: PROFILE_LEN=8192 (sequence length for ncu and nsys), SKIP_NCU=1, SKIP_NSYS=1
set -uo pipefail

TAG="${1:?usage: scripts/run_bench.sh <tag> \"<GPU name>\" [cuda_arch]}"
GPU_NAME="${2:?usage: scripts/run_bench.sh <tag> \"<GPU name>\" [cuda_arch]}"
ARCH="${3:-}"
PROFILE_LEN="${PROFILE_LEN:-8192}"

cd "$(dirname "$0")/.."
RES="results"
PROF="profiles/$TAG"
mkdir -p "$RES" "$PROF"

step() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; WARNINGS+=("$*"); }
WARNINGS=()

find_tool() {  # find_tool <name> <extra path>...
  local name="$1"; shift
  if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return 0; fi
  local p
  for p in "$@"; do [ -x "$p" ] && { echo "$p"; return 0; }; done
  return 1
}

# ---------------------------------------------------------------- 1. build
step "Building (arch: ${ARCH:-from CMakeLists})"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ${ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$ARCH} >/dev/null \
  && cmake --build build --target decode -j \
  || { echo "build failed, stopping"; exit 1; }

# ---------------------------------------------------------------- 2. GPU info
step "Recording GPU info"
{
  echo "tag: $TAG"
  echo "gpu: $GPU_NAME"
  date -u +"date: %Y-%m-%d %H:%M UTC"
  echo
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,clocks.max.graphics --format=csv
  echo
  nvcc --version | tail -n 2
  echo
  nvidia-smi
} > "$RES/${TAG}_gpu.txt" 2>&1
head -n 6 "$RES/${TAG}_gpu.txt"

# ---------------------------------------------------------------- 3. fixtures
step "Checking fixtures"
if [ ! -d fixtures/len_8192 ]; then
  echo "fixtures missing, generating them (needs torch in the active python)"
  python3 oracle/make_fixtures.py || { echo "could not generate fixtures, stopping"; exit 1; }
fi
ls fixtures

# ---------------------------------------------------------------- 4. benchmark
step "Benchmarking every kernel at every length (ceiling measured on this GPU)"
./build/decode --csv > "$RES/$TAG.csv" 2> "$RES/${TAG}_bench.log" \
  || warn "a kernel failed its correctness check, see $RES/$TAG.csv"
cat "$RES/${TAG}_bench.log"
column -s, -t "$RES/$TAG.csv" | cut -c1-170

# ---------------------------------------------------------------- 5. ncu
if [ -z "${SKIP_NCU:-}" ]; then
  step "Nsight Compute: one profile per kernel at L=${PROFILE_LEN}"
  if NCU="$(find_tool ncu /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu)"; then
    # performance counters are often admin-only: try plain first, then sudo
    RUN_NCU=("$NCU")
    if ! "$NCU" --set full -o "$PROF/_permtest" -f ./build/decode 1 --profile 100 >"$PROF/_permtest.log" 2>&1; then
      if grep -q ERR_NVGPUCTRPERM "$PROF/_permtest.log"; then
        echo "counters need admin rights, switching to sudo"
        RUN_NCU=(sudo env "PATH=$PATH" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "$NCU")
      fi
    fi
    rm -f "$PROF"/_permtest*

    ids=$(./build/decode --csv --len 1 --ceiling 1 2>/dev/null | awk -F, 'NR>1 {print $1 ":" $2}')
    for entry in $ids; do
      id="${entry%%:*}"; name="${entry#*:}"
      out="$PROF/$(printf '%02d' "$id")_${name}_${PROFILE_LEN}"
      echo "  kernel $id ($name)"
      "${RUN_NCU[@]}" --set full -f -o "$out" ./build/decode "$id" --profile "$PROFILE_LEN" >"$out.log" 2>&1 \
        || warn "ncu failed for kernel $id, see $out.log"
    done
  else
    warn "ncu not found; install nsight-compute and re-run with SKIP_NSYS=1"
  fi
fi

# ---------------------------------------------------------------- 6. nsys
if [ -z "${SKIP_NSYS:-}" ]; then
  step "Nsight Systems: timeline at L=${PROFILE_LEN}"
  if NSYS="$(find_tool nsys /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys)"; then
    "$NSYS" profile --trace=cuda --force-overwrite true -o "$PROF/nsys_all" \
      ./build/decode --bench --len "$PROFILE_LEN" >"$PROF/nsys_all.log" 2>&1 \
      || warn "nsys failed, see $PROF/nsys_all.log"
  else
    warn "nsys not found; install nsight-systems and re-run with SKIP_NCU=1"
  fi
fi

# ---------------------------------------------------------------- 7. bundle
step "Bundling"
sudo chown -R "$(id -u):$(id -g)" "$PROF" 2>/dev/null || true
BUNDLE="${TAG}_bundle.tar.gz"
tar czf "$BUNDLE" "$PROF" $(ls -d "$RES/$TAG".* "$RES/${TAG}_"* 2>/dev/null)
ls -lh "$BUNDLE"
echo "ncu reports: $(ls "$PROF"/*.ncu-rep 2>/dev/null | wc -l)   nsys reports: $(ls "$PROF"/*.nsys-rep 2>/dev/null | wc -l)"

if [ "${#WARNINGS[@]}" -gt 0 ]; then
  printf '\n%d warning(s):\n' "${#WARNINGS[@]}"; printf '  - %s\n' "${WARNINGS[@]}"
fi
