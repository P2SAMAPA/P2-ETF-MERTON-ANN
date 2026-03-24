"""
train_predict.py — P2-ETF-MERTON-ANN
Daily pipeline that processes one option (A or B) based on environment variable.
"""

import os
import sys
from datetime import datetime
from config import EQUITY_ETFS, EQUITY_REGIME, EQUITY_BENCHMARK, FI_ETFS, FI_REGIME, FI_BENCHMARK
from utils import process_module, save_signal_to_hf

def main():
    print("=" * 60)
    print("P2-ETF-MERTON-ANN: Daily Training & Prediction")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    option = os.environ.get("OPTION", "A")
    print(f"Running Option {option}")

    # Parameters (adjustable via environment)
    ETA = 0.5
    HORIZONS = [21, 63, 126]
    N_PATHS = int(os.environ.get("N_PATHS", "5000"))
    EPOCHS = int(os.environ.get("EPOCHS", "200"))

    # Process Equity
    print("\n>>> Starting Equity Module Processing <<<")
    equity_signal = process_module(
        "equity", EQUITY_ETFS, EQUITY_REGIME, option, HORIZONS, ETA, N_PATHS, EPOCHS
    )
    if not equity_signal or 'error' in equity_signal:
        print(f"✗ Equity processing failed: {equity_signal.get('error', 'Unknown error')}")
        sys.exit(1)
    print(f"Equity processing complete. Signal: {equity_signal.get('selected_etf', 'ERROR')}")
    save_signal_to_hf(equity_signal, "equity", option)

    # Process Fixed Income
    print("\n>>> Starting Fixed Income Module Processing <<<")
    fi_signal = process_module(
        "fixed_income", FI_ETFS, FI_REGIME, option, HORIZONS, ETA, N_PATHS, EPOCHS
    )
    if not fi_signal or 'error' in fi_signal:
        print(f"✗ Fixed Income processing failed: {fi_signal.get('error', 'Unknown error')}")
        sys.exit(1)
    print(f"Fixed Income processing complete. Signal: {fi_signal.get('selected_etf', 'ERROR')}")
    save_signal_to_hf(fi_signal, "fi", option)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
