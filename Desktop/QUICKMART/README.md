# Quickmart Forecast App

## Local run

1. Install dependencies:
   `pip install -r requirements.txt`
2. Generate the next-month forecast file:
   `python generate_next_month_forecast.py`
3. Start the Streamlit app:
   `streamlit run streamlit_app.py`

## What is included

This repository is set up as a code-only app project.
Large source datasets, trained model files, notebook files, and generated prediction CSVs are intentionally excluded from git.

To run the app, place the required Quickmart dataset file in the project folder:

- `QUICKMART_final_data.csv`

## Files

- `streamlit_app.py`: Streamlit interface
- `quickmart_forecast_lib.py`: reusable forecasting logic
- `generate_next_month_forecast.py`: creates the next-month prediction CSV used by the app
- `quickmart_monthly_forecast.py`: earlier training/evaluation script

## Expected workflow

1. Add `QUICKMART_final_data.csv` to the project folder.
2. Run `python generate_next_month_forecast.py`.
3. Run `streamlit run streamlit_app.py`.

## Online deployment

This app is ready for platforms like Streamlit Community Cloud.
Push this folder to GitHub, then create a new Streamlit app and set:

- Main file path: `streamlit_app.py`
- Python version: default platform version is fine

If you want, the next step can be adding charts or branding before deployment.
