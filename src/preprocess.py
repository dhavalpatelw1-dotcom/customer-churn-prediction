import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    df.drop("customerID", axis=1, inplace=True)

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    df = pd.get_dummies(
        df,
        drop_first=True
    )

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, scaler