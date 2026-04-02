from __future__ import annotations

from pathlib import Path
import json

from quickmart_forecast_lib import generate_next_month_forecast


PREDICTIONS_PATH = Path("quickmart_next_month_predictions.csv")
METRICS_PATH = Path("quickmart_next_month_metrics.json")


def main() -> None:
    predictions, metrics = generate_next_month_forecast()
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
