import pandas as pd
import os

C_FILES = [
    "500.perlbench_r.csv", "502.gcc_r.csv", "505.mcf_r.csv", "525.x264_r.csv",
    "507.cactuBSSN_r.csv", "519.lbm_r.csv", "538.imagick_r.csv",
    "544.nab_r.csv", "557.xz_r.csv"
]

FORT_FILES = [
    "548.exchange2_r.csv", "549.fotonik3d_r.csv", "554.roms_r.csv",
    "521.wrf_r.csv", "527.cam4_r.csv",
    "503.bwaves_r.csv"
]

CPP_FILES = [
    "520.omnetpp_r.csv", "523.xalancbmk_r.csv", "531.deepsjeng_r.csv",
    "541.leela_r.csv", "508.namd_r.csv", "510.parest_r.csv", "511.povray_r.csv", "526.blender_r.csv"
]


def main():
    dir = "./"
    for file in os.listdir(dir):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(dir, file)
        df = pd.read_csv(path)

        if file in CPP_FILES:
            df["Language"] = "cpp"
        elif file in FORT_FILES:
            df["Language"] = "fortran"
        elif file in C_FILES:
            df["Language"] = "c"
        else:
            print(f"Warning: {file} not found in any language list — skipping Language tag")
            continue

        total_executions = df["Executed"].sum()
        
        df["weight"] = df["Executed"] / total_executions
        df["rate"] = df["Taken"] / df["Executed"]

        df.to_csv(path, index=False)
        print(f"Tagged {file} as {df['Language'].iloc[0]}")


if __name__ == "__main__":
    main()