import joblib
import pandas as pd

model = joblib.load(
    "../models/churn_model.pkl"
)

scaler = joblib.load(
    "../models/scaler.pkl"
)

feature_names = joblib.load(
    "../models/feature_names.pkl"
)

sample_customer = pd.DataFrame(
    [[12, 75.5, 900.0] + [0]*(len(feature_names)-3)],
    columns=feature_names
)

sample_scaled = scaler.transform(
    sample_customer
)

prediction = model.predict(
    sample_scaled
)

prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0]

print("\nPrediction Result")
print("-" * 30)

if prediction == 1:
    print("Customer likely to churn")
else:
    print("Customer likely to stay")

print(f"Stay Probability : {probability[0]:.2%}")
print(f"Churn Probability: {probability[1]:.2%}")