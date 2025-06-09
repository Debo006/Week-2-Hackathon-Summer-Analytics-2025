import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load data
train_df = pd.read_csv("hacktrain.csv")
test_df = pd.read_csv("hacktest.csv")

# Drop unwanted column
train_df = train_df.drop(columns=["Unnamed: 0"])
test_ids = test_df["ID"]
test_df = test_df.drop(columns=["Unnamed: 0"])

# Separate features and labels
X_train = train_df.drop(columns=["ID", "class"])
y_train = train_df["class"]
X_test = test_df.drop(columns=["ID"])

# Impute missing values using column mean
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Feature scaling (important for some models, also standard practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Encode class labels into integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Define a more powerful model: Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)

# Optional: cross-validation accuracy check
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train_encoded, cv=5, scoring="accuracy")
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f}")

# Train on full data
rf_model.fit(X_train_scaled, y_train_encoded)

# Predict on test data
y_pred_encoded = rf_model.predict(X_test_scaled)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# Prepare submission
submission_df = pd.DataFrame({
    "ID": test_ids,
    "class": y_pred_labels
})

# Save to CSV
submission_df.to_csv("submission_rf.csv", index=False)
print("submission_rf.csv generated successfully!")
