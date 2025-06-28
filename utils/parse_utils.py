import pandas as pd

def parse_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame returned by yfinance to have standard OHLCV columns.
    Handles MultiIndex columns and single-ticker quirks.
    """
    if df is None or df.empty:
        return df

    # If MultiIndex columns, flatten to last level (field names)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # If all columns are the same (e.g., ['AAPL', 'AAPL', ...]), set to OHLCV
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if len(df.columns) == 5 and len(set(df.columns)) == 1:
        df.columns = expected_cols

    # If columns are not as expected, try to fix or warn
    if not all(col in df.columns for col in expected_cols):
        print('[parse_yf_df] Warning: Columns are not standard OHLCV:', df.columns)

    return df 