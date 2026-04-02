from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = Path("QUICKMART_final_data.csv")
MODEL_PATH = Path("quickmart_monthly_forecast_model.json")
QUANTITY_MODEL_PATH = Path("quickmart_monthly_quantity_forecast_model.json")
METRICS_PATH = Path("quickmart_monthly_forecast_metrics.json")
PREDICTIONS_PATH = Path("quickmart_monthly_forecast_predictions.csv")


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    monthly = df.groupby(["Date", "STORE_NAME_CLEAN", "ITEM_NAME"], as_index=False).agg(
        {
            "TOTAL SALES": "sum",
            "QUANTITY": "sum",
            "SUPPLIER": "first",
            "CATEGORY": "first",
            "DEPARTMENT": "first",
            "SUB DEPARTMENT": "first",
            "MICRO DEPARTMENT": "first",
            "HANDLER": "first",
            "Region": "first",
        }
    )

    all_months = pd.DataFrame({"Date": sorted(monthly["Date"].unique())})
    pairs = monthly[
        [
            "STORE_NAME_CLEAN",
            "ITEM_NAME",
            "SUPPLIER",
            "CATEGORY",
            "DEPARTMENT",
            "SUB DEPARTMENT",
            "MICRO DEPARTMENT",
            "HANDLER",
            "Region",
        ]
    ].drop_duplicates(subset=["STORE_NAME_CLEAN", "ITEM_NAME"])

    panel = pairs.merge(all_months, how="cross")
    panel = panel.merge(
        monthly,
        on=[
            "Date",
            "STORE_NAME_CLEAN",
            "ITEM_NAME",
            "SUPPLIER",
            "CATEGORY",
            "DEPARTMENT",
            "SUB DEPARTMENT",
            "MICRO DEPARTMENT",
            "HANDLER",
            "Region",
        ],
        how="left",
    )

    panel["TOTAL SALES"] = panel["TOTAL SALES"].fillna(0.0)
    panel["QUANTITY"] = panel["QUANTITY"].fillna(0.0)
    panel = panel.sort_values(["STORE_NAME_CLEAN", "ITEM_NAME", "Date"]).reset_index(drop=True)
    return panel


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    group_cols = ["STORE_NAME_CLEAN", "ITEM_NAME"]

    for lag in [1, 2, 3]:
        panel[f"sales_lag_{lag}"] = panel.groupby(group_cols)["TOTAL SALES"].shift(lag)
        panel[f"qty_lag_{lag}"] = panel.groupby(group_cols)["QUANTITY"].shift(lag)

    shifted_sales = panel.groupby(group_cols)["TOTAL SALES"].shift(1)
    panel["sales_roll_mean_3"] = shifted_sales.groupby(
        [panel["STORE_NAME_CLEAN"], panel["ITEM_NAME"]]
    ).transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    panel["sales_roll_std_3"] = shifted_sales.groupby(
        [panel["STORE_NAME_CLEAN"], panel["ITEM_NAME"]]
    ).transform(lambda s: s.rolling(window=3, min_periods=1).std())
    panel["sales_roll_std_3"] = panel["sales_roll_std_3"].fillna(0.0)
    panel["active_prev_month"] = (panel["sales_lag_1"] > 0).astype("int8")
    panel["month_number"] = panel["Date"].dt.month.astype("int8")
    panel["quarter"] = panel["Date"].dt.quarter.astype("int8")

    feature_ready = panel.dropna(subset=["sales_lag_1", "sales_lag_2", "sales_lag_3"]).copy()
    return feature_ready


def prepare_xy(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        "STORE_NAME_CLEAN",
        "ITEM_NAME",
        "SUPPLIER",
        "CATEGORY",
        "DEPARTMENT",
        "SUB DEPARTMENT",
        "MICRO DEPARTMENT",
        "HANDLER",
        "Region",
        "month_number",
        "quarter",
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_3",
        "qty_lag_1",
        "qty_lag_2",
        "qty_lag_3",
        "sales_roll_mean_3",
        "sales_roll_std_3",
        "active_prev_month",
    ]

    X = panel[feature_columns].copy()
    y = panel["TOTAL SALES"].copy()

    categorical_columns = [
        "STORE_NAME_CLEAN",
        "ITEM_NAME",
        "SUPPLIER",
        "CATEGORY",
        "DEPARTMENT",
        "SUB DEPARTMENT",
        "MICRO DEPARTMENT",
        "HANDLER",
        "Region",
    ]
    for column in categorical_columns:
        X[column] = X[column].astype("category")

    return X, y


def prepare_quantity_xy(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        "STORE_NAME_CLEAN",
        "ITEM_NAME",
        "SUPPLIER",
        "CATEGORY",
        "DEPARTMENT",
        "SUB DEPARTMENT",
        "MICRO DEPARTMENT",
        "HANDLER",
        "Region",
        "month_number",
        "quarter",
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_3",
        "qty_lag_1",
        "qty_lag_2",
        "qty_lag_3",
        "sales_roll_mean_3",
        "sales_roll_std_3",
        "active_prev_month",
    ]

    X = panel[feature_columns].copy()
    y = panel["QUANTITY"].copy()

    categorical_columns = [
        "STORE_NAME_CLEAN",
        "ITEM_NAME",
        "SUPPLIER",
        "CATEGORY",
        "DEPARTMENT",
        "SUB DEPARTMENT",
        "MICRO DEPARTMENT",
        "HANDLER",
        "Region",
    ]
    for column in categorical_columns:
        X[column] = X[column].astype("category")

    return X, y


def split_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = panel[panel["Date"] < "2026-01-01"].copy()
    valid_df = panel[panel["Date"] == "2026-01-01"].copy()
    test_df = panel[panel["Date"] == "2026-02-01"].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Expected non-empty train, validation, and test sets.")

    return train_df, valid_df, test_df


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=10,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def save_outputs(
    sales_model: XGBRegressor,
    quantity_model: XGBRegressor,
    metrics: dict,
    predictions: pd.DataFrame,
) -> None:
    sales_model.get_booster().save_model(MODEL_PATH)
    quantity_model.get_booster().save_model(QUANTITY_MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(PREDICTIONS_PATH, index=False)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    panel = build_monthly_panel(df)
    panel = add_lag_features(panel)
    train_df, valid_df, test_df = split_panel(panel)

    X_train, y_train = prepare_xy(train_df)
    X_valid, y_valid = prepare_xy(valid_df)
    X_test, y_test = prepare_xy(test_df)
    Xq_train, yq_train = prepare_quantity_xy(train_df)
    Xq_valid, yq_valid = prepare_quantity_xy(valid_df)
    Xq_test, yq_test = prepare_quantity_xy(test_df)

    model = train_model(X_train, y_train, X_valid, y_valid)
    quantity_model = train_model(Xq_train, yq_train, Xq_valid, yq_valid)

    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)
    valid_qty_pred = np.clip(quantity_model.predict(Xq_valid), a_min=0.0, a_max=None)
    test_qty_pred = np.clip(quantity_model.predict(Xq_test), a_min=0.0, a_max=None)

    metrics = {
        "dataset": str(DATA_PATH),
        "grain": "Date x STORE_NAME_CLEAN x ITEM_NAME",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "sales_target": "monthly_total_sales",
        "sales_best_iteration": int(model.best_iteration),
        "sales_validation_metrics": regression_metrics(y_valid, valid_pred),
        "sales_test_metrics": regression_metrics(y_test, test_pred),
        "quantity_target": "monthly_quantity",
        "quantity_best_iteration": int(quantity_model.best_iteration),
        "quantity_validation_metrics": regression_metrics(yq_valid, valid_qty_pred),
        "quantity_test_metrics": regression_metrics(yq_test, test_qty_pred),
    }

    predictions = test_df[
        [
            "Date",
            "STORE_NAME_CLEAN",
            "ITEM_NAME",
            "MICRO DEPARTMENT",
            "SUPPLIER",
            "QUANTITY",
            "TOTAL SALES",
            "sales_lag_1",
            "sales_lag_2",
            "sales_lag_3",
            "qty_lag_1",
            "qty_lag_2",
            "qty_lag_3",
        ]
    ].copy()
    predictions["predicted_total_sales"] = test_pred
    predictions["predicted_quantity"] = test_qty_pred
    predictions["absolute_error"] = (predictions["TOTAL SALES"] - predictions["predicted_total_sales"]).abs()
    predictions["quantity_absolute_error"] = (predictions["QUANTITY"] - predictions["predicted_quantity"]).abs()

    save_outputs(model, quantity_model, metrics, predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
