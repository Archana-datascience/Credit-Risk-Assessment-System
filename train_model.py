import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

print("Loading dataset...")

df = pd.read_csv("credit_risk_dataset.csv")

target = "credit_risk"

X = df.drop(columns=[target])
y = df[target]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)
prob = model.predict_proba(X_test_scaled)[:,1]

accuracy = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, prob)

cv_auc = cross_val_score(
    model,
    scaler.transform(X),
    y,
    cv=5,
    scoring="roc_auc"
)

artifacts = {
    "model": model,
    "scaler": scaler,
    "feature_names": feature_names,
    "categorical_cols": categorical_cols,
    "label_encoders": label_encoders,
    "model_performance": {
        "accuracy": accuracy,
        "auc_score": auc,
        "cv_auc_mean": cv_auc.mean(),
        "cv_auc_std": cv_auc.std()
    }
}

joblib.dump(
    artifacts,
    "credit_risk_model.pkl"
)

print("\nModel Saved Successfully")
print("credit_risk_model.pkl created")

print(f"Accuracy : {accuracy:.4f}")
print(f"AUC      : {auc:.4f}")