#!/usr/bin/env python3
import os
import re
import logging
import glob
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Configuration -----
ROOT_DIR = '/Users/mrsmoothy/Downloads/bilançolar'
TARGET_SHEET_NAMES = [
    "Bilanço",
    "Gelir Tablosu (Yıllıklan.)",
    "Nakit Akış (Yıllıklan.)"
]
FILE_PATTERN = re.compile(r".*\([A-Z]{3}\).*\.xlsx$", re.IGNORECASE)

# ----- Extraction Helper Functions -----
def parse_numeric(value, scale=1):
    """
    Convert a cell value to a float.
    - Handles Turkish decimal format (comma as decimal separator)
    - Recognizes negative values denoted by parentheses.
    - Applies a scale factor if a note like 'bin' or 'milyon' is detected.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        neg = False
        if value.startswith('(') and value.endswith(')'):
            neg = True
            value = value[1:-1]
        value = value.replace('.', '').replace(',', '.')
        try:
            num = float(value)
        except Exception:
            return np.nan
        if neg:
            num = -num
        return num * scale
    elif isinstance(value, (int, float)):
        return value * scale
    else:
        return np.nan

def detect_scale_factor(df):
    """
    Look at the first few rows for hints that values are scaled (e.g., 'bin' for thousands, 'milyon' for millions).
    Returns the detected scale factor (default 1 if none is detected).
    """
    scale = 1
    for idx in range(min(5, len(df))):
        row = df.iloc[idx].astype(str).str.lower().tolist()
        for cell in row:
            if "bin" in cell:
                return 1000
            elif "milyon" in cell:
                return 1e6
    return scale

def detect_header_row(df):
    """
    Heuristically determine the header row by searching for date-like patterns.
    Returns the index of the header row.
    """
    date_pattern = re.compile(r'\d{1,2}[\/\.]\d{1,2}[\/\.]\d{2,4}')
    for i in range(len(df)):
        row = df.iloc[i].astype(str)
        if any(date_pattern.search(str(cell)) for cell in row):
            return i
    return 0

def parse_financial_sheet(df):
    """
    Process a raw worksheet DataFrame:
      - Detects and sets the header row.
      - Drops header rows from the data region.
      - Detects a scale factor and applies it to numeric values.
      - Adds a 'level' column based on indentation of the account names.
    """
    header_row_idx = detect_header_row(df)
    df.columns = df.iloc[header_row_idx]
    df = df.drop(index=range(0, header_row_idx+1)).reset_index(drop=True)
    
    scale = detect_scale_factor(df)
    
    # Assume first column holds account descriptions.
    def get_level(account):
        if isinstance(account, str):
            return len(account) - len(account.lstrip())
        return 0

    df.rename(columns={df.columns[0]: 'account'}, inplace=True)
    df['level'] = df['account'].apply(get_level)
    
    # Normalize numeric columns (skip 'account' and 'level')
    for col in df.columns:
        if col not in ['account', 'level']:
            df[col] = df[col].apply(lambda x: parse_numeric(x, scale))
    
    return df

def process_excel_file(file_path):
    """
    Load an Excel file and process each target worksheet.
    Returns a dictionary mapping sheet names to parsed DataFrames.
    """
    logging.info(f"Processing file: {file_path}")
    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return {}
    
    parsed_sheets = {}
    for sheet_name in TARGET_SHEET_NAMES:
        if sheet_name in xls.sheet_names:
            try:
                df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                parsed_df = parse_financial_sheet(df_raw)
                parsed_sheets[sheet_name] = parsed_df
            except Exception as e:
                logging.error(f"Error processing sheet {sheet_name} in file {file_path}: {e}")
        else:
            logging.warning(f"Sheet {sheet_name} not found in file {file_path}")
    return parsed_sheets

def traverse_and_process(root_dir):
    """
    Recursively traverse the given root directory and process each Excel file matching the target pattern.
    Returns a dictionary with file paths as keys and sheet-data dictionaries as values.
    """
    all_data = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if FILE_PATTERN.match(filename):
                file_path = os.path.join(dirpath, filename)
                company_data = process_excel_file(file_path)
                if company_data:
                    all_data[file_path] = company_data
    return all_data

# ----- Trend Calculation Helper Functions -----
def get_trend_series(df, account_substring, use_regex=False, context=""):
    """
    Search the 'account' column for rows that contain the given substring.
    If use_regex is True, treat account_substring as a regex pattern; otherwise, escape it.
    Returns a pandas Series with datetime indices (parsed from period columns in 'YYYY/MM' format)
    and the sum of corresponding values as data.
    The 'context' parameter is used to log additional information about the source (e.g., company name).
    """
    import numpy as np
    import pandas as pd
    import logging

    if use_regex:
        pattern = account_substring
    else:
        pattern = re.escape(account_substring)
    mask = df['account'].str.contains(pattern, case=False, na=False)
    if not mask.any():
        logging.info(f"{context} - No matching rows for pattern '{account_substring}'.")
        return pd.Series(dtype=float)
    matched = df[mask]
    trend_data = {}
    for col in df.columns:
        if col in ['account', 'level']:
            continue
        try:
            # Parse column header as date (e.g., "2024/12")
            date_val = pd.to_datetime(col, format="%Y/%m", errors='raise')
            value = matched[col].sum()
            if np.isnan(value) or np.isinf(value):
                logging.warning(f"{context} - Column '{col}' for account pattern '{account_substring}' produced an invalid sum ({value}). This time point will be omitted.")
                continue
            trend_data[date_val] = value
        except Exception as e:
            logging.error(f"{context} - Error processing column '{col}' for account pattern '{account_substring}': {e}")
            continue
    if not trend_data:
        return pd.Series(dtype=float)
    trend_series = pd.Series(trend_data).sort_index()
    return trend_series

def compute_trend_ratio(numerator_series, denominator_series):
    """
    Compute a ratio as a time series by dividing numerator_series by denominator_series,
    aligning on common dates.
    """
    common_index = numerator_series.index.intersection(denominator_series.index)
    if common_index.empty:
        return pd.Series(dtype=float)
    ratio_series = numerator_series[common_index] / denominator_series[common_index]
    return ratio_series

def compute_ratios_for_company(sheets):
    """
    Compute trend-based financial ratios as time series for a given company.
    Returns a dictionary where each ratio name maps to a pandas Series.
    """
    ratios = {}
    balance_df = sheets.get("Bilanço")
    income_df = sheets.get("Gelir Tablosu (Yıllıklan.)")
    cash_df = sheets.get("Nakit Akış (Yıllıklan.)")
    
    if income_df is not None:
        brut_kar_series = get_trend_series(income_df, "Brüt Kar (Zarar)")
        satis_gelir_series = get_trend_series(income_df, "Satış Gelirleri")
        ratios["Brüt Kar/Hasılat"] = compute_trend_ratio(brut_kar_series, satis_gelir_series)
        
        genel_yonetim_series = get_trend_series(income_df, "Genel Yönetim Giderleri")
        ratios["Yönetim/Kar"] = compute_trend_ratio(genel_yonetim_series, brut_kar_series)
        
        pazarlama_series = get_trend_series(income_df, "Pazarlama, Satış ve Dağıtım Giderleri")
        ratios["Pazarlama/Kar"] = compute_trend_ratio(pazarlama_series, brut_kar_series)
        
        arge_series = get_trend_series(income_df, "Araştırma ve Geliştirme Giderleri")
        ratios["Arge/Kar"] = compute_trend_ratio(arge_series, brut_kar_series)
        
        amortisman_series = get_trend_series(income_df, "Amortisman")
        ratios["Amortisman/Kar"] = compute_trend_ratio(amortisman_series, brut_kar_series)
        
        donem_net_series = get_trend_series(balance_df, "Dönem\s*Net\s*Kar(?:/Zararı)?", use_regex=True)
        ratios["Dönem Net/Brüt Kar"] = compute_trend_ratio(donem_net_series, brut_kar_series)
    else:
        logging.warning("Income Statement data not available for trend ratio calculations.")
    
    if balance_df is not None:
        toplam_varlik_series = get_trend_series(balance_df, "Toplam Varlıklar")
        donem_net_series = get_trend_series(balance_df, "Dönem\s*Net\s*Kar(?:/Zararı)?", use_regex=True)
        ratios["Dönem Net/Toplam Varlık"] = compute_trend_ratio(donem_net_series, toplam_varlik_series)
    else:
        logging.warning("Balance Sheet or Income Statement missing for Dönem Net/Toplam Varlık trend ratio.")
    
    if balance_df is not None:
        finansal_borc_series = get_trend_series(balance_df, "Finansal Borçlar")
        ortaklik_series = get_trend_series(balance_df, "Ana Ortaklığa Ait Özkaynak")
        ratios["Finansal Borç/Özkaynak"] = compute_trend_ratio(finansal_borc_series, ortaklik_series)
    else:
        logging.warning("Balance Sheet data missing for Finansal Borç/Özkaynak trend ratio.")
    
    if cash_df is not None and balance_df is not None:
        yatirim_nakit_series = get_trend_series(cash_df, "Yatırım Faaliyetlerinden Kaynaklanan Nakit Akışları")
        donem_net_series = get_trend_series(balance_df, "Dönem\s*Net\s*Kar(?:/Zararı)?", use_regex=True)
        ratios["Sermaye Harcaması/Dönem Net"] = compute_trend_ratio(yatirim_nakit_series, donem_net_series)
    else:
        logging.warning("Cash Flow or Balance Sheet missing for Sermaye Harcaması/Dönem Net trend ratio.")
    
    if balance_df is not None:
        gecmis_yil_series = get_trend_series(balance_df, "Geçmiş Yıllar Kar/Zararları")
        yedek_series = get_trend_series(balance_df, "Kardan Ayrılan Kısıtlanmış Yedekler")
        ratios["Geçmiş Yıl Kar + Yedek Akçe"] = gecmis_yil_series.add(yedek_series, fill_value=0)
    else:
        logging.warning("Balance Sheet data missing for Geçmiş Yıl Kar + Yedek Akçe trend ratio.")
    
    if balance_df is not None:
        donem_net_series = get_trend_series(balance_df, "Dönem\s*Net\s*Kar(?:/Zararı)?", use_regex=True)
        odenmis_sermaye_series = get_trend_series(balance_df, "Ödenmiş Sermaye")
        ratios["Hisse Başı Kazanç"] = compute_trend_ratio(donem_net_series, odenmis_sermaye_series)
    else:
        logging.warning("Income Statement or Balance Sheet missing for Hisse Başı Kazanç trend ratio.")
    
    if cash_df is not None and balance_df is not None:
        yatirim_nakit_series = get_trend_series(cash_df, "Yatırım Faaliyetlerinden Kaynaklanan Nakit Akışları")
        odenmis_sermaye_series = get_trend_series(balance_df, "Ödenmiş Sermaye")
        ratios["Hisse Başı Sermaye Harcaması"] = compute_trend_ratio(yatirim_nakit_series, odenmis_sermaye_series)
    else:
        logging.warning("Cash Flow or Balance Sheet missing for Hisse Başı Sermaye Harcaması trend ratio.")
    
    return ratios

# ----- Main Execution -----
def main():
    logging.info("Starting extraction of financial statements.")
    extracted_data = traverse_and_process(ROOT_DIR)
    logging.info(f"Extraction complete. Processed {len(extracted_data)} files.")
    
    results = {}
    for file_path, sheets in extracted_data.items():
        company_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nProcessing company: {company_name}")
        ratios = compute_ratios_for_company(sheets)
        results[company_name] = ratios
        for ratio_name, series in ratios.items():
            print(f"\nTrend for {ratio_name} for {company_name}:")
            print(series)
    
if __name__ == "__main__":
    main()