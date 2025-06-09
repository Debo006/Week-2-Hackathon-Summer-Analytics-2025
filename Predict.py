import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("hacktrain.csv")
test_df = pd.read_csv("hacktest.csv")

# Drop unnecessary columns
train_df.drop(columns=["Unnamed: 0", "ID"], inplace=True)
test_ids = test_df["ID"]
test_df.drop(columns=["Unnamed: 0", "ID"], inplace=True)

# Separate labels
y = train_df["class"]
X = train_df.drop(columns=["class"])

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(test_df)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/Test split for evaluation
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred_val = clf.predict(X_val)
print(classification_report(y_val, y_pred_val, target_names=label_encoder.classes_))

# Predict on test set
y_test_pred = clf.predict(X_test_imputed)
predicted_labels = label_encoder.inverse_transform(y_test_pred)

# Output predictions
submission = pd.DataFrame({
    "ID": test_ids,
    "Predicted": predicted_labels
})
submission.to_csv("submission.csv", index=False)
