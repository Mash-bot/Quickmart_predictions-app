from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PREDICTIONS_PATH = Path("quickmart_next_month_predictions.csv")
METRICS_PATH = Path("quickmart_next_month_metrics.json")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["Date"])


@st.cache_data
def load_metrics() -> dict:
    import json

    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def apply_filters(
    df: pd.DataFrame,
    month_label: str,
    product: str,
    supplier: str,
    store: str,
    micro_department: str,
) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[filtered["forecast_month_label"] == month_label]

    if product != "All":
        filtered = filtered[filtered["ITEM_NAME"] == product]
    if supplier != "All":
        filtered = filtered[filtered["SUPPLIER"] == supplier]
    if store != "All":
        filtered = filtered[filtered["STORE_NAME_CLEAN"] == store]
    if micro_department != "All":
        filtered = filtered[filtered["MICRO DEPARTMENT"] == micro_department]

    return filtered


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title="Quickmart Forecast App", layout="wide")

st.title("Quickmart Next-Month Forecast")
st.caption("Predicts next-month quantity and total sales at product level, with supplier breakdowns and downloadable results.")

if not PREDICTIONS_PATH.exists() or not METRICS_PATH.exists():
    st.error(
        "Forecast files are missing. Run `python generate_next_month_forecast.py` first to create the next-month predictions."
    )
    st.stop()

predictions = load_predictions()
metrics = load_metrics()

month_options = sorted(predictions["forecast_month_label"].dropna().unique().tolist())
product_options = ["All"] + sorted(predictions["ITEM_NAME"].dropna().unique().tolist())
supplier_options = ["All"] + sorted(predictions["SUPPLIER"].dropna().unique().tolist())
store_options = ["All"] + sorted(predictions["STORE_NAME_CLEAN"].dropna().unique().tolist())
micro_options = ["All"] + sorted(predictions["MICRO DEPARTMENT"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filters")
    month_label = st.selectbox("Month", month_options, index=0)
    product = st.selectbox("Product", product_options, index=0)
    supplier = st.selectbox("Supplier", supplier_options, index=0)
    store = st.selectbox("Store", store_options, index=0)
    micro_department = st.selectbox("Micro Department", micro_options, index=0)

filtered = apply_filters(predictions, month_label, product, supplier, store, micro_department)

summary_sales = filtered["predicted_total_sales"].sum()
summary_quantity = filtered["predicted_quantity"].sum()
row_count = len(filtered)

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Sales", f"{summary_sales:,.0f}")
col2.metric("Predicted Quantity", f"{summary_quantity:,.0f}")
col3.metric("Matching Product Rows", f"{row_count:,}")

st.write(
    f"Forecast month: `{metrics['forecast_month_label']}`. "
    f"Sales validation R²: `{metrics['sales_validation_metrics']['r2']:.3f}`. "
    f"Quantity validation R²: `{metrics['quantity_validation_metrics']['r2']:.3f}`."
)

if filtered.empty:
    st.warning("No predictions match the current filters.")
    st.stop()

supplier_breakdown = (
    filtered.groupby(["SUPPLIER", "ITEM_NAME"], as_index=False)[
        ["predicted_total_sales", "predicted_quantity", "predicted_quantity_rounded"]
    ]
    .sum()
    .sort_values(["predicted_total_sales", "predicted_quantity"], ascending=False)
)

product_breakdown = (
    filtered.groupby(
        ["ITEM_NAME", "SUPPLIER", "MICRO DEPARTMENT", "STORE_NAME_CLEAN"],
        as_index=False,
    )[["predicted_total_sales", "predicted_quantity", "predicted_quantity_rounded"]]
    .sum()
    .sort_values(["predicted_total_sales", "predicted_quantity"], ascending=False)
)

st.subheader("Filtered Forecast Detail")
st.dataframe(
    filtered[
        [
            "Date",
            "STORE_NAME_CLEAN",
            "ITEM_NAME",
            "MICRO DEPARTMENT",
            "SUPPLIER",
            "predicted_total_sales",
            "predicted_quantity",
            "predicted_quantity_rounded",
        ]
    ].sort_values(["predicted_total_sales", "predicted_quantity"], ascending=False),
    use_container_width=True,
    height=420,
)

st.download_button(
    "Download Filtered Forecast CSV",
    data=to_csv_bytes(filtered),
    file_name="quickmart_filtered_forecast.csv",
    mime="text/csv",
)

st.subheader("Supplier Product Breakdown")
st.dataframe(supplier_breakdown, use_container_width=True, height=320)
st.download_button(
    "Download Supplier Breakdown CSV",
    data=to_csv_bytes(supplier_breakdown),
    file_name="quickmart_supplier_breakdown.csv",
    mime="text/csv",
)

st.subheader("Product Breakdown")
st.dataframe(product_breakdown, use_container_width=True, height=320)
st.download_button(
    "Download Product Breakdown CSV",
    data=to_csv_bytes(product_breakdown),
    file_name="quickmart_product_breakdown.csv",
    mime="text/csv",
)
