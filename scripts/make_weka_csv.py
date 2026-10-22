import pandas as pd

df = pd.read_csv("data/processed/windows.csv")

classes = [
    "Normal",
    "External Position",
    "Altitude",
    "Mechanical",
    "Global Position",
]

stats = ("__mean", "__std", "__min", "__max")
features = [
    col for col in df.columns
    if col.endswith(stats)
]

df = df[df["label"].isin(classes)]
result = df[features + ["label"]].copy()

result.to_csv(
    "data/processed/weka_windows.csv",
    index=False,
    encoding="utf-8"
)

print("Weka dosyasi hazir!")
print("Pencere sayisi:", len(result))
print("Ozellik sayisi:", len(features))
print("Sinif dagilimi:")
print(result["label"].value_counts())
