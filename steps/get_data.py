import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")

print("Path to dataset files:", path)

for file in os.listdir(path):
    full_path = os.path.join(path, file)

    df = pd.read_csv(full_path)
    df.to_csv("../data/win_quality_red.csv")


def get_data():
    return pd.read_csv(full_path)