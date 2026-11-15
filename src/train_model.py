import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def generate_housing_dataset(samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generates a synthetic Indian housing dataset with features and target prices."""
    rng = np.random.default_rng(seed)
    bedrooms = rng.integers(1, 6, size=samples)
    min_area = np.maximum(450, bedrooms * 220)
    area_sqft = np.round(rng.uniform(min_area, min_area + 2400), 1)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, size=samples), 1, 6)
    age_years = rng.integers(0, 31, size=samples)

    # Realistic pricing logic (in INR Lakhs)
    price_lakhs = (
        (area_sqft * 0.055)
        + (bedrooms * 6.5)
        + (bathrooms * 3.0)
        - (age_years * 0.45)
        + rng.normal(0, 4.0, size=samples)
    )
    price_lakhs = np.round(np.maximum(price_lakhs, 15.0), 2)

    return pd.DataFrame({
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "price_lakhs": price_lakhs,
    })


def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "housing_training_data.csv")
    model_path = os.path.join(base_dir, "house_price_model.joblib")

    # 1. Generate and save dataset
    df = generate_housing_dataset(samples=1000)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} training rows -> {csv_path}")

    # 2. Split features and target
    X = df[["area_sqft", "bedrooms", "bathrooms", "age_years"]].to_numpy()
    y = df["price_lakhs"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate and save serialized model
    preds = model.predict(X_test)
    print(f"Model R2 Score: {r2_score(y_test, preds):.4f}")
    print(f"Mean Absolute Error: ₹{mean_absolute_error(y_test, preds):.2f} Lakhs")

    joblib.dump(model, model_path)
    print(f"Saved trained model -> {model_path}")


if __name__ == "__main__":
    main()