# train.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Create a tiny demo dataset (you can replace with your CSV later)
data = {
    "job_title": ["Junior Developer","Senior Developer","Data Scientist","Manager","Junior Developer","Senior Developer"],
    "years_experience": [1, 8, 4, 10, 2, 7],
    "location": ["Bengaluru","San Francisco","Bengaluru","New York","Mumbai","San Francisco"],
    "education_level": ["Bachelors","Bachelors","Masters","MBA","Bachelors","Masters"],
    "remote_ratio": [50, 20, 100, 0, 70, 10],
    "salary": [400000, 2400000, 1200000, 3000000, 450000, 2000000]  # annual salary in INR or arbitrary units
}
df = pd.DataFrame(data)

# Features and target
X = df.drop("salary", axis=1)
y = df["salary"]

# Preprocessing: one-hot for categorical columns, pass numeric columns
categorical_cols = ["job_title", "location", "education_level"]
numeric_cols = ["years_experience", "remote_ratio"]

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough"  # leave numeric cols as-is
)

# Pipeline: preprocess -> model
pipe = Pipeline([
    ("pre", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split (small)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

# Print a tiny evaluation
from sklearn.metrics import mean_absolute_error
preds = pipe.predict(X_test)
print("MAE on test:", mean_absolute_error(y_test, preds))

# Save pipeline
joblib.dump(pipe, "model.joblib")
print("Saved model to model.joblib")
