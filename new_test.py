#!/usr/bin/env python3
import os
import re
import logging
import pandas as pd
import numpy as np

# ----- Configuration -----
ROOT_DIR = '/Users/mrsmoothy/Downloads/bilançolar'
TARGET_SHEET_NAMES = [
    "Bilanço",
    "Gelir Tablosu (Yıllıktan.)",
    "Nakit Akış (Yıllıktan.)"
]
FILE_PATTERN = re.compile(r".*\([A-Z]{3}\).*\.xlsx$", re.IGNORECASE)

# ----- Logging Configuration -----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Helper Functions -----
def parse_numeric(value, scale=1):
    """
    Convert a cell value to a float.
    - Handles Turkish decimal format (comma as decimal separator)
    - Recognizes negative values represented by parentheses.
    - Applies a scale factor if a note like 'bin' (thousands) or 'milyon' (millions) is detected.
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
    Examine the first few rows for scale hints (e.g., 'bin' or 'milyon').
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
    Financial statements often have columns with period dates (e.g., 31/12/2020).
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
      - Adds a 'level' column to preserve hierarchical relationships based on indentation.
      - Normalizes numeric cells.
    """
    # Detect header row using date patterns
    header_row_idx = detect_header_row(df)
    df.columns = df.iloc[header_row_idx]
    df = df.drop(index=range(0, header_row_idx+1)).reset_index(drop=True)
    
    # Detect if a scale factor (e.g., thousands or millions) is mentioned
    scale = detect_scale_factor(df)
    
    # Assume first column contains the account descriptions.
    # Create a 'level' column by counting leading spaces in account names.
    def get_level(account):
        if isinstance(account, str):
            return len(account) - len(account.lstrip())
        return 0

    df.rename(columns={df.columns[0]: 'account'}, inplace=True)
    df['level'] = df['account'].apply(get_level)
    
    # Normalize numeric columns
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
                logging.error(f"Error processing sheet '{sheet_name}' in file {file_path}: {e}")
        else:
            logging.warning(f"Sheet '{sheet_name}' not found in file {file_path}")
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

# ----- Main Execution -----
def main():
    logging.info("Starting the extraction of financial statements.")
    extracted_data = traverse_and_process(ROOT_DIR)
    
    # Export each processed sheet to a CSV file
    for file_path, sheets in extracted_data.items():
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        for sheet_name, df in sheets.items():
            csv_filename = f"{base_name}_{sheet_name.replace(' ', '_')}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Exported {csv_filename} with shape: {df.shape}")
    
    logging.info("Extraction and export completed.")

if __name__ == "__main__":
    main()