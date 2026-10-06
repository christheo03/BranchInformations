#!/usr/bin/env bash
set -euo pipefail

export PIN_ROOT=/home/students/cs/2022/ctheod03/Desktop/ADE/pin-external-3.31-98869-gfa6f126a8-gcc-linux
export PY_LOC=/home/students/cs/2022/ctheod03/Desktop/ADE/py_scripts
export SDE_ROOT=/home/students/cs/2022/ctheod03/Desktop/ADE/sde-external-10.8.0-2026-03-15-lin

RESULTS_DIR=$(pwd)

case "${1:-}" in
    521)
        LOC="521.wrf_r"
        ARGS=()
        EXE="wrf_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="rsl.out.0000"
        ;;
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
    525)
        LOC="525.x264_r"
        ARGS=(
            "--pass"
            "1"
            "--stats"
            "x264_stats.log" 
            "--bitrate" 
            "1000" 
            "--frames" 
            "1000" 
            "-o" 
            "BuckBunny_New.264" 
            "BuckBunny.yuv" 
            "1280x720"
            )
        EXE="x264_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="run_000-1000_x264_r_base.EPL221SingleCore-gcc4.8.5-linux3.10-bits-64_x264_pass1.out"
        ;;
    526)
        LOC="526.blender_r"
        ARGS=(
            "sh3_no_char.blend" 
            "--render-output" 
            "sh3_no_char_" 
            "--threads" 
            "1" 
            "-b" 
            "-F" 
            "RAWTGA" 
            "-s" 
            "849" 
            "-e" 
            "849" 
            "-a"
            )
        EXE="blender_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="sh3_no_char.849.spec.out"
        ;;
    523)
        LOC="523.xalancbmk_r"
        ARGS=(
            "-v"
            "t5.xml"
            "xalanc.xsl"
            )
        EXE="cpuxalan_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="ref-t5.out"
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
        ARGS=("bwaves_1")
        IN="bwaves_1.in"
        OUT="bwaves_1.out"
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
    527)
        LOC="527.cam4_r"
        ARGS=()
        EXE="cam4_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="cam4_r_base.EPL221SingleCore-gcc4.8.5-linux3.10-bits-64.txt"
        ;;
    541)
        LOC="541.leela_r"
        ARGS=(
            "ref.sgf"
        )
        EXE="leela_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="ref.out"
        ;;
    544)
        LOC="544.nab_r"
        ARGS=(
            "1am0" 
            "1122214447"
            "122"
        )
        EXE="nab_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="1am0.out"
        ;;
    548)
        LOC="548.exchange2_r"
        ARGS=(
            "6"
        )
        EXE="exchange2_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="exchange2.txt"
        ;;
    549)
        LOC="549.fotonik3d_r"
        ARGS=()
        EXE="fotonik3d_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="fotonik3d_r.log"
        ;;
    554)
        LOC="554.roms_r"
        ARGS=()
        IN="ocean_benchmark2.in.x"
        EXE="roms_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="ocean_benchmark2.log"
        ;;
    557)
        LOC="557.xz_r"
        ARGS=(
            "cld.tar.xz" 
            "160" 
            "19cf30ae51eddcbefda78dd06014b4b96281456e078ca7c13e1c0c9e6aaea8dff3efb4ad6b0456697718cede6bd5454852652806a657bb56e07d61128434b474" 
            "59796407" 
            "61004416" 
            "6"
        )
        EXE="xz_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="cld.tar-160-6.out"
        ;;
    531)
        LOC="531.deepsjeng_r"
        ARGS=(
            "ref.txt"
        )
        EXE="deepsjeng_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
        OUT="ref.out"
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
    538)
        LOC="538.imagick_r"
        ARGS=(
            "-limit"
            "disk" 
            "0" 
            "refrate_input.tga" 
            "-edge" 
            "41" 
            "-resample" 
            "181%" 
            "-emboss" 
            "31" 
            "-colorspace" 
            "YUV" 
            "-mean-shift" 
            "19x19+15%" 
            "-resize" 
            "30%" 
            "refrate_output.tga"
        )
        OUT="refrate_convert.out"
        EXE="imagick_r_base.EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64"
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
        echo "Usage: $0 {500|502|503|505|507|508|510|511|519|520|521|523|525|526|531|538|541|544|548|549}"
        exit 1
        ;;
esac

BASE="/home/students/cs/2022/ctheod03/Desktop/ADE/SPECprog/${LOC}/run/run_base_refrate_EPL221SingleCore-gcc4.8.5-static-linux3.10-bits-64.0000"

echo "Running benchmark inside: $BASE"

(
    cd "$BASE"
    IN="${IN:-/dev/null}"
    OUT="${OUT:-/dev/null}"
    echo "[1/2] Running PIN Tool Branch Analyser"
    time "$PIN_ROOT/pin" \
        -t "$PIN_ROOT/source/tools/MyPinTool/obj-intel64/faster.so" \
        -- \
        "./$EXE" "${ARGS[@]}" < "${IN}" > "${OUT}"

    echo "[2/2] Generate and extract CFG information"
    time python3 "${PY_LOC}/angrversion.py" "${EXE}"
)

echo "Moving results to $RESULTS_DIR"

mv "$BASE/branches.csv" "$RESULTS_DIR/${LOC}.csv" 2>/dev/null || true

echo "Done. Final CSV: ${LOC}.csv"
