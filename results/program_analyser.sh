#!/usr/bin/env bash
set -euo pipefail

export PIN_ROOT=/home/students/cs/2022/ctheod03/Desktop/ADE/pin-external-3.31-98869-gfa6f126a8-gcc-linux
export PY_LOC=/home/students/cs/2022/ctheod03/Desktop/ADE/py_scripts
export SDE_ROOT=/home/students/cs/2022/ctheod03/Desktop/ADE/sde-external-10.8.0-2026-03-15-lin

RESULTS_DIR=$(pwd)

case "${1:-}" in
    500)
        LOC="500.perlbench_r"
        ARGS=(
            "-I./lib"
            "checkspam.pl"
            "2500"
            "5"
            "25"
            "11"
            "150"
            "1"
            "1"
            "1"
            "1"
        )
        EXE="perlbench_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    505)
        LOC="505.mcf_r"
        ARGS=("inp.in")
        EXE="mcf_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    511)
        LOC="511.povray_r"
        ARGS=("SPEC-benchmark-ref.ini")
        EXE="povray_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    503)
        LOC="503.bwaves_r"
        ARGS=(
            "bwaves_1"
            "<"
            "bwaves_1.in"
            )
        EXE="bwaves_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    520)
        LOC="520.omnetpp_r"
        ARGS=("-c" "General" "-r" "0")
        EXE="omnetpp_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    507)
        LOC="507.cactuBSSN_r"
        ARGS=(
            "spec_ref.par"
        )
        EXE="cactusBSSN_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    508)
        LOC="508.namd_r"
        ARGS=(
            "--input" 
            "apoa1.input" 
            "--output" 
            "apoa1.ref.output" 
            "--iterations" 
            "65"
        )
        EXE="namd_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    510)
        LOC="510.parest_r"
        ARGS=(
            "ref.prm" 
        )
        EXE="parest_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    519)
        LOC="519.lbm_r"
        ARGS=(
            "3000" 
            "reference.dat" 
            "0" 
            "0" 
            "100_100_130_ldc.of" 
        )
        EXE="lbm_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    502)
        LOC="502.gcc_r"
        ARGS=(
            "gcc-pp.c"
            "-O3"
            "-finline-limit=0"
            "-fif-conversion"
            "-fif-conversion2"
            "-o"
            "gcc-pp.opts-O3_-finline-limit_0_-fif-conversion_-fif-conversion2.s"
        )
        EXE="cpugcc_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        ;;
    *)
        echo "Usage: $0 {500|502|503|505|507|511|520}"
        exit 1
        ;;
esac

BASE="/home/students/cs/2022/ctheod03/Desktop/ADE/SPECprog/${LOC}/run/run_base_refrate_EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64.0000"

echo "Running benchmark inside: $BASE"

(
    cd "$BASE"

    echo "[1/5] Running PIN Tool Branch Analyser"
    time "$PIN_ROOT/pin" \
        -t "$PIN_ROOT/source/tools/MyPinTool/obj-intel64/faster.so" \
        -- \
        "./$EXE" "${ARGS[@]}"

    echo "[2/5] Producing the dcfg graph"
    time "$SDE_ROOT/sde64" \
        -dcfg 1 \
        -dcfg:analysis 1 \
        -dcfg:cfg 1 \
        -dcfg:source 0 \
        -dcfg:symbols 0 \
        -dcfg:write_debug 0 \
        -dcfg:write_trace 0 \
        -dcfg:read_dcfg_if_avail 0 \
        -dcfg:out_base_name test \
        -- \
        "./$EXE" "${ARGS[@]}"

    bunzip2 -f "test.dcfg.json.bz2"

    echo "[3/5] Extract information from dcfg"
    time python3 "${PY_LOC}/get_dcfg_heuristics.py"

    rm -f "test.dcfg.json"

    echo "[4/5] Running the PIN Tool Registers_Behavior"
    time "$PIN_ROOT/pin" \
        -t "$PIN_ROOT/source/tools/MyPinTool/obj-intel64/UBD_NU.so" \
        -csv "branches.csv" \
        -- \
        "./$EXE" "${ARGS[@]}"

    echo "[5/5] Merge informations to a single csv"
    time python3 "${PY_LOC}/merge_csv.py"
    rm -f "ubd_results.csv"
)

echo "Moving results to $RESULTS_DIR"

mv "$BASE/branches.csv" "$RESULTS_DIR/${LOC}.csv" 2>/dev/null || true

echo "Done. Final CSV: ${LOC}.csv"
