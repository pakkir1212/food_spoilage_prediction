import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def evaluate(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "food_spoilage.csv")
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    features = [
        "Preservative",
        "VacuumPress",
        "ContaminationLevel",
        "CoolTemp",
    ]
    X = df[features]
    y = df["Spoilage"]

    preprocessor = ColumnTransformer([
        ("preservative", OneHotEncoder(handle_unknown="ignore"),
         ["Preservative"]),
        ("numeric", "passthrough",
         ["VacuumPress", "ContaminationLevel", "CoolTemp"]),
    ])

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42,
        ),
    }

    report = {
        "dataset_rows": int(len(df)),
        "target": "Spoilage",
        "evaluation": "Leave-One-Out Cross-Validation",
        "models": {},
        "note": "Preliminary evaluation on 16 experimental runs; not evidence of real-world generalization.",
    }

    table = pd.DataFrame({"Actual_Spoilage": y.to_numpy()})

    for name, estimator in models.items():
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ])
        pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
        report["models"][name] = evaluate(y, pred)
        table[name.replace(" ", "_") + "_Prediction"] = pred

    table.to_csv(results_dir / "predictions.csv", index=False)

    with open(results_dir / "baseline_metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
