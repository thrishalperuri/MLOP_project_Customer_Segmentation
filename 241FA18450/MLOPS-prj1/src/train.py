import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from preprocess import prepare_features


# Load training data
train_df = pd.read_csv("../data/raw/Train.csv")

# Features
numerical_features = [
    "Age",
    "Work_Experience",
    "Family_Size"
]

categorical_features = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score",
    "Var_1"
]

# Target
y = train_df["Segmentation"]

# Prepare features
X = prepare_features(
    train_df,
    numerical_features,
    categorical_features
)

# Split into training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(
    model,
    "../models/logistic_regression_model.pkl"
)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

print("Model trained and saved successfully.")