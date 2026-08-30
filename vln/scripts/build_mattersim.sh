#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: vln/scripts/build_mattersim.sh [--check]

Build the pinned Matterport3DSimulator Python module used by DUET, HAMT, and
GOAT on the fixed NavTTA server layout.  The default action downloads missing
source files, builds the CPython 3.8 extension, and verifies all three discrete
VLN environments.  --check only verifies that the expected extension exists.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

CHECK_ONLY=0
case "${1:-}" in
    '') ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
esac
[[ "$#" -le 1 ]] || die "too many arguments"

REPO_ROOT=/data1/wxy/code/NavTTA
VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln
ENV_ROOT="${VLN_ROOT}/envs"
TMP_ROOT="${VLN_ROOT}/tmp"
NATIVE_ROOT="${ENV_ROOT}/mattersim-native"
NATIVE_BIN="${NATIVE_ROOT}/bin"
NATIVE_INCLUDE="${NATIVE_ROOT}/include"
NATIVE_LIB="${NATIVE_ROOT}/lib"
NATIVE_CMAKE="${NATIVE_BIN}/cmake"
NATIVE_PKG_CONFIG="${NATIVE_BIN}/pkg-config"
NATIVE_CC="${NATIVE_BIN}/x86_64-conda-linux-gnu-cc"
NATIVE_CXX="${NATIVE_BIN}/x86_64-conda-linux-gnu-c++"
OPENCV_DIR="${NATIVE_LIB}/cmake/opencv4"
JSONCPP_PC="${NATIVE_LIB}/pkgconfig/jsoncpp.pc"
SIM_ROOT="${REPO_ROOT}/vln/data/simulators/Matterport3DSimulator"
BUILD_ROOT="${SIM_ROOT}/build"
DUET_PYTHON="${ENV_ROOT}/duet/bin/python"
MATTERSIM_REV=589d091b111333f9e9f9d6cfd021b2eb68435925
PYBIND11_REV=86e2ad4f77442c3350f9a2476650da6bee253c52
BUILD_STAMP=.navtta-build-revisions
BUILDER_CONTRACT=2

[[ -x "${DUET_PYTHON}" ]] || die "missing DUET Python: ${DUET_PYTHON}"
EXT_SUFFIX="$(
    "${DUET_PYTHON}" -c \
        'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")'
)"
MODULE="${BUILD_ROOT}/MatterSim${EXT_SUFFIX}"
mkdir -p "${TMP_ROOT}"
command -v flock >/dev/null 2>&1 || die "missing command: flock"
exec {BUILD_LOCK_FD}>"${TMP_ROOT}/mattersim-build.lock"
if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    flock -s -n "${BUILD_LOCK_FD}" || die "MatterSim build is in progress"
else
    flock -n "${BUILD_LOCK_FD}" || die "another MatterSim build is in progress"
fi

check_import() {
    local environment="$1"
    local build_root="${2:-${BUILD_ROOT}}"
    local python="${ENV_ROOT}/${environment}/bin/python"
    [[ -x "${python}" ]] || die "missing ${environment} Python: ${python}"
    [[ -f "${build_root}/MatterSim${EXT_SUFFIX}" ]] || return 1
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${build_root}" \
    LD_LIBRARY_PATH="${build_root}:${NATIVE_LIB}:${ENV_ROOT}/${environment}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    MATTERSIM_EXPECTED="${build_root}/MatterSim${EXT_SUFFIX}" \
        "${python}" - <<'PY'
import MatterSim
import os
from pathlib import Path

simulator = MatterSim.Simulator()
simulator.setRenderingEnabled(False)
actual = Path(MatterSim.__file__).resolve()
expected = Path(os.environ["MATTERSIM_EXPECTED"]).resolve()
if actual != expected:
    raise SystemExit("unexpected MatterSim module: {} != {}".format(actual, expected))
print("MatterSim import OK:", MatterSim.__file__)
PY
}

check_stamp() {
    local build_root="$1"
    local expected actual
    expected="$(printf 'mattersim=%s\npybind11=%s\npython_ext_suffix=%s\nbuilder_contract=%s\nnative_prefix=%s' \
        "${MATTERSIM_REV}" "${PYBIND11_REV}" "${EXT_SUFFIX}" \
        "${BUILDER_CONTRACT}" "${NATIVE_ROOT}")"
    [[ -f "${build_root}/${BUILD_STAMP}" ]] || return 1
    actual="$(<"${build_root}/${BUILD_STAMP}")"
    [[ "${actual}" == "${expected}" ]]
}

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    [[ -f "${MODULE}" ]] || die \
        "missing MatterSim module: ${MODULE}; run vln/scripts/build_mattersim.sh"
    check_stamp "${BUILD_ROOT}" || die \
        "MatterSim build is unpinned or stale; run vln/scripts/build_mattersim.sh"
    for environment in duet hamt goat; do
        check_import "${environment}" || die \
            "MatterSim import failed in ${environment}; rebuild it"
    done
    printf 'MatterSim module ready: %s\n' "${MODULE}"
    exit 0
fi

[[ -x "${NATIVE_CMAKE}" ]] || die \
    "missing native CMake: ${NATIVE_CMAKE}; install the mattersim-native environment"
[[ -x "${NATIVE_PKG_CONFIG}" ]] || die \
    "missing native pkg-config: ${NATIVE_PKG_CONFIG}"
[[ -x "${NATIVE_CC}" ]] || die "missing native C compiler: ${NATIVE_CC}"
[[ -x "${NATIVE_CXX}" ]] || die "missing native C++ compiler: ${NATIVE_CXX}"
[[ -f "${OPENCV_DIR}/OpenCVConfig.cmake" ]] || die \
    "missing native OpenCV CMake package: ${OPENCV_DIR}/OpenCVConfig.cmake"
[[ -f "${JSONCPP_PC}" ]] || die \
    "missing native JsonCpp pkg-config file: ${JSONCPP_PC}"
[[ -f "${NATIVE_INCLUDE}/json/json.h" ]] || die \
    "missing native JsonCpp header: ${NATIVE_INCLUDE}/json/json.h"
[[ -f "${NATIVE_INCLUDE}/crypt.h" ]] || die \
    "missing native libxcrypt header: ${NATIVE_INCLUDE}/crypt.h"
for command in curl tar cp mv; do
    command -v "${command}" >/dev/null 2>&1 || die "missing command: ${command}"
done

mkdir -p "${SIM_ROOT}"
STAGE="$(mktemp -d "${TMP_ROOT}/mattersim-build.XXXXXX")"
NEW_BUILD="${SIM_ROOT}/.build.new.$$"
PREVIOUS_BUILD="${SIM_ROOT}/.build.previous.$$"
HAD_PREVIOUS_BUILD=0
DEPLOYMENT_ACTIVE=0
cleanup() {
    local status="$?"
    trap - EXIT INT TERM
    if [[ "${DEPLOYMENT_ACTIVE}" -eq 1 ]]; then
        if [[ "${HAD_PREVIOUS_BUILD}" -eq 1 && \
              ( -e "${PREVIOUS_BUILD}" || -L "${PREVIOUS_BUILD}" ) ]]; then
            [[ ! -e "${BUILD_ROOT}" && ! -L "${BUILD_ROOT}" ]] || \
                rm -rf -- "${BUILD_ROOT}"
            mv "${PREVIOUS_BUILD}" "${BUILD_ROOT}" || true
        elif [[ "${HAD_PREVIOUS_BUILD}" -eq 0 ]]; then
            [[ ! -e "${BUILD_ROOT}" && ! -L "${BUILD_ROOT}" ]] || \
                rm -rf -- "${BUILD_ROOT}"
        fi
    fi
    rm -rf -- "${STAGE}"
    [[ ! -e "${NEW_BUILD}" && ! -L "${NEW_BUILD}" ]] || \
        rm -rf -- "${NEW_BUILD}"
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

download_archive() {
    local url="$1"
    local output="$2"
    curl --http1.1 -fL \
        --retry 8 --retry-all-errors --retry-delay 3 \
        --connect-timeout 20 --max-time 900 \
        "${url}" -o "${output}"
}

printf 'Preparing Matterport3DSimulator %s\n' "${MATTERSIM_REV}"
download_archive \
    "https://codeload.github.com/peteanderson80/Matterport3DSimulator/tar.gz/${MATTERSIM_REV}" \
    "${STAGE}/mattersim.tar.gz"
tar -xzf "${STAGE}/mattersim.tar.gz" -C "${STAGE}"
SOURCE_ROOT="${STAGE}/Matterport3DSimulator-${MATTERSIM_REV}"
[[ -f "${SOURCE_ROOT}/CMakeLists.txt" ]] || \
    die "downloaded Matterport3DSimulator archive has an unexpected layout"

download_archive \
    "https://codeload.github.com/pybind/pybind11/tar.gz/${PYBIND11_REV}" \
    "${STAGE}/pybind11.tar.gz"
tar -xzf "${STAGE}/pybind11.tar.gz" -C "${STAGE}"
PYBIND11_ROOT="${STAGE}/pybind11-${PYBIND11_REV}"
[[ -f "${PYBIND11_ROOT}/CMakeLists.txt" ]] || \
    die "downloaded pybind11 archive has an unexpected layout"
mkdir -p "${SOURCE_ROOT}/pybind11"
cp -a "${PYBIND11_ROOT}/." "${SOURCE_ROOT}/pybind11/"

"${DUET_PYTHON}" - \
    "${SOURCE_ROOT}/src/lib/NavGraph.cpp" \
    "${SOURCE_ROOT}/include/NavGraph.hpp" <<'PY'
from pathlib import Path
import sys

replacements = {
    Path(sys.argv[1]): ("CV_LOAD_IMAGE_ANYDEPTH", "cv::IMREAD_ANYDEPTH"),
    Path(sys.argv[2]): ("<jsoncpp/json/json.h>", "<json/json.h>"),
}
for path, (old, new) in replacements.items():
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit("unexpected compatibility token count in {}: {}".format(path, old))
    path.write_text(text.replace(old, new), encoding="utf-8")
PY

export CMAKE_PREFIX_PATH="${NATIVE_ROOT}"
export PKG_CONFIG="${NATIVE_PKG_CONFIG}"
export PKG_CONFIG_PATH="${NATIVE_LIB}/pkgconfig:${NATIVE_ROOT}/share/pkgconfig"
export PKG_CONFIG_LIBDIR="${PKG_CONFIG_PATH}"
export CPPFLAGS="-I${NATIVE_INCLUDE}"
export CFLAGS="-I${NATIVE_INCLUDE}"
export CXXFLAGS="-I${NATIVE_INCLUDE}"
export LDFLAGS="-L${NATIVE_LIB} -Wl,-rpath,${NATIVE_LIB}"
export LIBRARY_PATH="${NATIVE_LIB}"
export CC="${NATIVE_CC}"
export CXX="${NATIVE_CXX}"
export PATH="${ENV_ROOT}/duet/bin:${NATIVE_BIN}:${PATH}"
export LD_LIBRARY_PATH="${NATIVE_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
STAGED_BUILD="${SOURCE_ROOT}/build"
"${NATIVE_CMAKE}" -S "${SOURCE_ROOT}" -B "${STAGED_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${NATIVE_ROOT}" \
    -DCMAKE_C_COMPILER="${NATIVE_CC}" \
    -DCMAKE_CXX_COMPILER="${NATIVE_CXX}" \
    -DCMAKE_BUILD_RPATH="${NATIVE_LIB}" \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_INSTALL_RPATH="${NATIVE_LIB}" \
    -DPKG_CONFIG_EXECUTABLE="${NATIVE_PKG_CONFIG}" \
    -DOpenCV_DIR="${OPENCV_DIR}" \
    -DEGL_RENDERING=OFF \
    -DOSMESA_RENDERING=OFF \
    -DPYTHON_EXECUTABLE="${DUET_PYTHON}"
"${NATIVE_CMAKE}" --build "${STAGED_BUILD}" \
    --target MatterSimPython --parallel "${MAX_JOBS:-4}"

STAGED_MODULE="${STAGED_BUILD}/MatterSim${EXT_SUFFIX}"
[[ -f "${STAGED_MODULE}" ]] || \
    die "build finished without expected module: ${STAGED_MODULE}"
printf 'mattersim=%s\npybind11=%s\npython_ext_suffix=%s\nbuilder_contract=%s\nnative_prefix=%s\n' \
    "${MATTERSIM_REV}" "${PYBIND11_REV}" "${EXT_SUFFIX}" \
    "${BUILDER_CONTRACT}" "${NATIVE_ROOT}" \
    >"${STAGED_BUILD}/${BUILD_STAMP}"

for environment in duet hamt goat; do
    check_import "${environment}" "${STAGED_BUILD}" || \
        die "staged MatterSim import failed in ${environment}"
done

# Preserve the upstream source and licenses beside the ignored native build.
cp -a \
    "${SOURCE_ROOT}/CMakeLists.txt" \
    "${SOURCE_ROOT}/LICENSE" \
    "${SOURCE_ROOT}/README.md" \
    "${SOURCE_ROOT}/cmake" \
    "${SOURCE_ROOT}/include" \
    "${SOURCE_ROOT}/src" \
    "${SIM_ROOT}/"
mkdir -p "${SIM_ROOT}/pybind11"
cp -a "${SOURCE_ROOT}/pybind11/." "${SIM_ROOT}/pybind11/"

# Build outside the live tree, then replace the complete build directory only
# after all three Python environments have imported the staged extension.
[[ ! -e "${NEW_BUILD}" ]] || die "temporary deployment path exists: ${NEW_BUILD}"
cp -a "${STAGED_BUILD}" "${NEW_BUILD}"
if [[ -e "${BUILD_ROOT}" || -L "${BUILD_ROOT}" ]]; then
    [[ ! -e "${PREVIOUS_BUILD}" ]] || \
        die "temporary backup path exists: ${PREVIOUS_BUILD}"
    HAD_PREVIOUS_BUILD=1
    DEPLOYMENT_ACTIVE=1
    mv "${BUILD_ROOT}" "${PREVIOUS_BUILD}"
else
    DEPLOYMENT_ACTIVE=1
fi
if ! mv "${NEW_BUILD}" "${BUILD_ROOT}"; then
    die "could not install the staged MatterSim build"
fi
if ! check_stamp "${BUILD_ROOT}"; then
    die "installed MatterSim revision stamp is invalid"
fi
for environment in duet hamt goat; do
    if ! check_import "${environment}"; then
        die "installed MatterSim import failed in ${environment}"
    fi
done
if [[ "${HAD_PREVIOUS_BUILD}" -eq 1 ]]; then
    rm -rf -- "${PREVIOUS_BUILD}"
fi
DEPLOYMENT_ACTIVE=0

printf 'MatterSim build ready: %s\n' "${MODULE}"
