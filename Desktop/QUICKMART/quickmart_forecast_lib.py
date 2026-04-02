from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = Path("QUICKMART_final_data.csv")
CATEGORICAL_COLUMNS = [
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
FEATURE_COLUMNS = [
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


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    monthly = data.groupby(["Date", "STORE_NAME_CLEAN", "ITEM_NAME"], as_index=False).agg(
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
    feature_panel = panel.copy()
    group_cols = ["STORE_NAME_CLEAN", "ITEM_NAME"]

    for lag in [1, 2, 3]:
        feature_panel[f"sales_lag_{lag}"] = feature_panel.groupby(group_cols)["TOTAL SALES"].shift(lag)
        feature_panel[f"qty_lag_{lag}"] = feature_panel.groupby(group_cols)["QUANTITY"].shift(lag)

    shifted_sales = feature_panel.groupby(group_cols)["TOTAL SALES"].shift(1)
    feature_panel["sales_roll_mean_3"] = shifted_sales.groupby(
        [feature_panel["STORE_NAME_CLEAN"], feature_panel["ITEM_NAME"]]
    ).transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    feature_panel["sales_roll_std_3"] = shifted_sales.groupby(
        [feature_panel["STORE_NAME_CLEAN"], feature_panel["ITEM_NAME"]]
    ).transform(lambda s: s.rolling(window=3, min_periods=1).std())
    feature_panel["sales_roll_std_3"] = feature_panel["sales_roll_std_3"].fillna(0.0)
    feature_panel["active_prev_month"] = (feature_panel["sales_lag_1"] > 0).astype("int8")
    feature_panel["month_number"] = feature_panel["Date"].dt.month.astype("int8")
    feature_panel["quarter"] = feature_panel["Date"].dt.quarter.astype("int8")

    return feature_panel.dropna(subset=["sales_lag_1", "sales_lag_2", "sales_lag_3"]).copy()


def prepare_features(panel: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = panel[FEATURE_COLUMNS].copy()
    y = panel[target].copy()

    for column in CATEGORICAL_COLUMNS:
        X[column] = X[column].astype("category")

    return X, y


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


def build_next_month_feature_frame(feature_ready: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    latest_date = feature_ready["Date"].max()
    next_month = latest_date + pd.offsets.MonthBegin(1)
    latest_rows = feature_ready[feature_ready["Date"] == latest_date].copy()

    next_rows = latest_rows[
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
            "TOTAL SALES",
            "QUANTITY",
            "sales_lag_1",
            "sales_lag_2",
            "qty_lag_1",
            "qty_lag_2",
        ]
    ].copy()

    next_rows["Date"] = next_month
    next_rows["month_number"] = next_month.month
    next_rows["quarter"] = next_month.quarter
    next_rows["sales_lag_1"] = latest_rows["TOTAL SALES"].to_numpy()
    next_rows["sales_lag_2"] = latest_rows["sales_lag_1"].to_numpy()
    next_rows["sales_lag_3"] = latest_rows["sales_lag_2"].to_numpy()
    next_rows["qty_lag_1"] = latest_rows["QUANTITY"].to_numpy()
    next_rows["qty_lag_2"] = latest_rows["qty_lag_1"].to_numpy()
    next_rows["qty_lag_3"] = latest_rows["qty_lag_2"].to_numpy()

    sales_window = np.column_stack(
        [
            latest_rows["TOTAL SALES"].to_numpy(),
            latest_rows["sales_lag_1"].to_numpy(),
            latest_rows["sales_lag_2"].to_numpy(),
        ]
    )
    next_rows["sales_roll_mean_3"] = sales_window.mean(axis=1)
    next_rows["sales_roll_std_3"] = sales_window.std(axis=1)
    next_rows["active_prev_month"] = (next_rows["sales_lag_1"] > 0).astype("int8")
    return next_rows, next_month


def generate_next_month_forecast(data_path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(data_path)
    panel = build_monthly_panel(df)
    feature_ready = add_lag_features(panel)

    latest_date = feature_ready["Date"].max()
    train_df = feature_ready[feature_ready["Date"] < latest_date].copy()
    valid_df = feature_ready[feature_ready["Date"] == latest_date].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Not enough history to train and validate the next-month forecast models.")

    X_train_sales, y_train_sales = prepare_features(train_df, "TOTAL SALES")
    X_valid_sales, y_valid_sales = prepare_features(valid_df, "TOTAL SALES")
    sales_model = train_model(X_train_sales, y_train_sales, X_valid_sales, y_valid_sales)

    X_train_qty, y_train_qty = prepare_features(train_df, "QUANTITY")
    X_valid_qty, y_valid_qty = prepare_features(valid_df, "QUANTITY")
    quantity_model = train_model(X_train_qty, y_train_qty, X_valid_qty, y_valid_qty)

    next_rows, next_month = build_next_month_feature_frame(feature_ready)
    X_next_sales, _ = prepare_features(next_rows.assign(**{"TOTAL SALES": 0.0}), "TOTAL SALES")
    X_next_qty, _ = prepare_features(next_rows.assign(**{"QUANTITY": 0.0}), "QUANTITY")

    predictions = next_rows[
        [
            "Date",
            "STORE_NAME_CLEAN",
            "ITEM_NAME",
            "MICRO DEPARTMENT",
            "SUPPLIER",
            "Region",
            "sales_lag_1",
            "sales_lag_2",
            "sales_lag_3",
            "qty_lag_1",
            "qty_lag_2",
            "qty_lag_3",
        ]
    ].copy()
    predictions["predicted_total_sales"] = sales_model.predict(X_next_sales)
    predictions["predicted_quantity"] = np.clip(quantity_model.predict(X_next_qty), a_min=0.0, a_max=None)
    predictions["predicted_quantity_rounded"] = predictions["predicted_quantity"].round().astype(int)
    predictions["forecast_month_label"] = pd.to_datetime(predictions["Date"]).dt.strftime("%B %Y")

    metrics = {
        "dataset": str(data_path),
        "forecast_month": next_month.strftime("%Y-%m-%d"),
        "forecast_month_label": next_month.strftime("%B %Y"),
        "grain": "Date x STORE_NAME_CLEAN x ITEM_NAME",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "sales_validation_metrics": regression_metrics(
            y_valid_sales,
            sales_model.predict(X_valid_sales),
        ),
        "quantity_validation_metrics": regression_metrics(
            y_valid_qty,
            np.clip(quantity_model.predict(X_valid_qty), a_min=0.0, a_max=None),
        ),
    }

    return predictions, metrics
