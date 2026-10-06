import pandas as pd
import os
def main():
    dir = "./"
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            
            df = pd.read_csv(file)

            print(df["weight"].sum())

if __name__ == "__main__":
    main()