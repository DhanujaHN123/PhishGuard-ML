
import json
import pandas as pd
import numpy as np
from urllib.parse import urlparse

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

FEATURE_FILE = "phishguard_features.csv"
DATASET_FILE = "PhiUSIIL_Phishing_URL_Dataset.csv"

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "subdomain_count",
    "digit_count",
    "digit_ratio",
    "special_count",
    "hyphen_count",
    "at_count",
    "query_parameter_count",
    "dot_count",
    "path_segments",
    "has_https",
    "has_ip",
    "has_suspicious_keyword",
    "keyword_count",
    "entropy",
    "has_punycode",
    "has_fragment"
]


def extract_hostname(value):
    url = str(value).strip()

    if "](" in url and url.endswith(")"):
        url = url.split("](", 1)[1][:-1]

    if "://" not in url:
        url = "https://" + url

    return (urlparse(url).hostname or "").lower().rstrip(".")


print("Loading processed features and original URLs...")

features = pd.read_csv(FEATURE_FILE)
original = pd.read_csv(DATASET_FILE)

if len(features) != len(original):
    raise ValueError(
        "Row counts differ. The processed features may not align "
        "with the original dataset. Do not train until this is resolved."
    )

if "URL" not in original.columns or "label" not in original.columns:
    raise ValueError("Original dataset must contain URL and label columns.")

if features.columns.tolist() != FEATURE_NAMES + ["label"]:
    raise ValueError("Processed CSV feature names or order do not match Model A.")

# Verify that the processed labels match the original labels.
expected_labels = (original["label"].astype(int) == 0).astype(int)

if not np.array_equal(
    features["label"].astype(int).to_numpy(),
    expected_labels.to_numpy()
):
    raise ValueError("Processed labels do not align with original labels.")

X = features[FEATURE_NAMES].copy()
y = features["label"].astype(int).copy()
groups = original["URL"].apply(extract_hostname)

if (groups == "").any():
    raise ValueError("Some URLs have missing hostnames.")

print("Dataset shape:", features.shape)
print("Feature count:", len(FEATURE_NAMES))
print("Splitting by hostname...")

# Reserve 20% of hostnames for the final test set.
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_val_idx, test_idx = next(
    split1.split(X, y, groups=groups)
)

# Split the remaining data into training and validation sets.
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=43
)

train_rel, val_rel = next(
    split2.split(
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups=groups.iloc[train_val_idx]
    )
)

train_idx = train_val_idx[train_rel]
val_idx = train_val_idx[val_rel]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

train_hosts = set(groups.iloc[train_idx])
val_hosts = set(groups.iloc[val_idx])
test_hosts = set(groups.iloc[test_idx])

assert not (train_hosts & val_hosts)
assert not (train_hosts & test_hosts)
assert not (val_hosts & test_hosts)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))
print("Hostname overlaps: 0")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

print("\nTraining Model A...")
model.fit(X_train, y_train)

val_pred = model.predict(X_val)

print("\n--- Validation Results ---")
print("Accuracy:", round(accuracy_score(y_val, val_pred) * 100, 3), "%")
print("Precision:", round(precision_score(y_val, val_pred) * 100, 3), "%")
print("Recall:", round(recall_score(y_val, val_pred) * 100, 3), "%")
print("F1-score:", round(f1_score(y_val, val_pred) * 100, 3), "%")

test_pred = model.predict(X_test)

print("\n--- Final Test Results: Model A ---")
print("Accuracy:", round(accuracy_score(y_test, test_pred) * 100, 3), "%")
print("Phishing precision:", round(precision_score(y_test, test_pred) * 100, 3), "%")
print("Phishing recall:", round(recall_score(y_test, test_pred) * 100, 3), "%")
print("Phishing F1:", round(f1_score(y_test, test_pred) * 100, 3), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_pred,
    labels=[0, 1],
    target_names=["Legitimate", "Phishing"],
    zero_division=0
))

# Export the model parameters for browser-side inference.
scaler = model.named_steps["scaler"]
classifier = model.named_steps["classifier"]

model_export = {
    "model_name": "PhishGuard AI",
    "model_type": "LogisticRegression",
    "feature_count": len(FEATURE_NAMES),
    "feature_names": FEATURE_NAMES,
    "classes": classifier.classes_.tolist(),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coefficients": classifier.coef_.tolist(),
    "intercept": classifier.intercept_.tolist()
}

with open("phishguard_model_final.json", "w", encoding="utf-8") as f:
    json.dump(model_export, f, indent=2)

print("\nExported phishguard_model_final.json")
print("Training and evaluation completed.")
