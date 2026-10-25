
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load Dataset
df = pd.read_csv("data/ai4i2020.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# 2. Select Features
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = pd.get_dummies(df[features], columns=["Type"])
y = df["Machine failure"]

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# 4. Train Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2
)

model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\n===== CONFUSION MATRIX =====")
print("Rows = Actual, Columns = Predicted")
print(cm)

# 8. Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance_df.to_string(index=False))

# 9. Save Model and Columns
joblib.dump(model, "motor_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

# Save evaluation results for Streamlit dashboard
evaluation_results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "feature_importance": importance_df.to_dict(orient="records")
}

joblib.dump(evaluation_results, "evaluation_results.pkl")

print("\nModel and evaluation results saved successfully!")

