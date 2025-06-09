import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Load data
train_df = pd.read_csv("hacktrain.csv")
test_df = pd.read_csv("hacktest.csv")

# Preprocessing
train_df.drop(columns=["Unnamed: 0"], inplace=True)
test_ids = test_df["ID"]
test_df.drop(columns=["Unnamed: 0"], inplace=True)
X_train = train_df.drop(columns=["ID", "class"])
y_train = train_df["class"]
X_test = test_df.drop(columns=["ID"])

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Encode target
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Best GB model
model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
cv_score = cross_val_score(model, X_train_scaled, y_train_encoded, cv=5, scoring='accuracy').mean()
print("Cross-Validation Accuracy:", cv_score)

# Fit and predict
model.fit(X_train_scaled, y_train_encoded)
y_pred_encoded = model.predict(X_test_scaled)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# Submission
submission = pd.DataFrame({"ID": test_ids, "class": y_pred_labels})
submission.to_csv("submission_best.csv", index=False)
print("submission_best.csv created!")

