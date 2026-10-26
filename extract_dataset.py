
import pandas as pd
from feature_extraction import extract_features

# Load the original dataset
input_file = "PhiUSIIL_Phishing_URL_Dataset.csv"

print("Loading original dataset...")
df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# Verify required columns
required_columns = {"URL", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Required columns {required_columns} not found. "
        "Check the actual column names before continuing."
    )

# Keep only rows with a URL and a valid label
df = df.dropna(subset=["URL", "label"]).copy()
df = df[df["label"].isin([0, 1])].copy()

# Extract the same 18 features for every URL
print("\nExtracting URL features...")

feature_rows = [
    extract_features(url)
    for url in df["URL"]
]

features_df = pd.DataFrame(feature_rows)

# Convert original labels:
# Original dataset: 1 = legitimate, 0 = phishing
# Our ML target:    0 = legitimate, 1 = phishing
features_df["label"] = (df["label"].astype(int) == 0).astype(int)

# Save processed data
output_file = "phishguard_features.csv"
features_df.to_csv(output_file, index=False)

print("\nFeature extraction completed!")
print("Processed dataset shape:", features_df.shape)
print("Saved to:", output_file)

print("\nML label distribution:")
print(features_df["label"].value_counts().rename(
    index={0: "Legitimate", 1: "Phishing"}
))



