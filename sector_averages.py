#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
from gpt_all_in_one import traverse_and_process, compute_ratios_for_company, ROOT_DIR

def get_company_ratio_values(ratios):
    """
    Given the dictionary of ratio trends for a company,
    extract a representative value for each ratio (using the last available data point).
    Returns a dict: {ratio_name: value}
    """
    rep = {}
    for ratio, series in ratios.items():
        if not series.empty:
            # Use the last available data point (assuming the series is sorted by date)
            rep[ratio] = series.iloc[-1]
    return rep

def main():
    # Traverse the folder structure and compute ratios for each company
    extracted_data = traverse_and_process(ROOT_DIR)
    
    # Data structure to hold company data grouped by sector:
    # { sector: { company: {ratio: value, ...}, ... } }
    sector_data = {}
    
    # Process each file from the extraction results
    for file_path, sheets in extracted_data.items():
        # Derive company name (without extension) and sector (parent folder name)
        company_name = os.path.splitext(os.path.basename(file_path))[0]
        sector = os.path.basename(os.path.dirname(file_path))
        ratios = compute_ratios_for_company(sheets)
        rep_values = get_company_ratio_values(ratios)
        if not rep_values:
            continue  # Skip companies with no valid ratios
        if sector not in sector_data:
            sector_data[sector] = {}
        sector_data[sector][company_name] = rep_values
    
    # Dictionary to hold consolidated sector averages
    all_sector_avg = {}
    
    # Now, for each sector, compute sectoral averages and rank companies by "Brüt Kar/Hasılat"
    for sector, companies in sector_data.items():
        print(f"\nSector: {sector}")
        # DataFrame: rows=companies, columns=ratio names
        df = pd.DataFrame.from_dict(companies, orient="index")
        # Compute average values for each ratio (ignoring NaN)
        sector_avg = df.mean(skipna=True)
        all_sector_avg[sector] = sector_avg
        print("\nSector Averages:")
        print(sector_avg)
        
        # Ranking companies by "Brüt Kar/Hasılat" (if available)
        if "Brüt Kar/Hasılat" in df.columns:
            # Higher value is assumed to be better
            top_companies = df["Brüt Kar/Hasılat"].dropna().sort_values(ascending=False).head(5)
            print("\nTop 5 Companies by Brüt Kar/Hasılat:")
            print(top_companies)
        else:
            print("\n'Brüt Kar/Hasılat' ratio not available for ranking in this sector.")
        
        # Save individual sector data
        output_csv = os.path.join("output_visuals", f"{sector}_sector_averages.csv")
        os.makedirs("output_visuals", exist_ok=True)
        df.to_csv(output_csv)
        print(f"Saved company ratios for sector '{sector}' to {output_csv}")
    
    # Consolidate all sector averages into one DataFrame
    consolidated_df = pd.DataFrame(all_sector_avg).T  # sectors as rows, ratios as columns
    print("\nConsolidated Sector Averages:")
    print(consolidated_df)
    consolidated_csv = os.path.join("output_visuals", "consolidated_sector_averages.csv")
    consolidated_df.to_csv(consolidated_csv)
    print(f"Saved consolidated sector averages to {consolidated_csv}")

if __name__ == "__main__":
    main()