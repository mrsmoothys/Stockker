#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from gpt_all_in_one import traverse_and_process, compute_ratios_for_company, ROOT_DIR

# Set up logging (logs to console and file)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('sector_trends.log', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def aggregate_trend_series(series_list, context=""):
    """
    Given a list of pandas Series (each with datetime index), align them on the union of all dates,
    and return the mean series (ignoring NaN values).
    The optional 'context' parameter is included in any logging messages to indicate the source of the data.
    """
    import numpy as np
    import pandas as pd
    import logging

    if not series_list:
        return pd.Series(dtype=float)
    # Concatenate along columns; join on outer to capture all dates.
    df = pd.concat(series_list, axis=1, join='outer')
    
    # Optionally, log a summary of the raw data before averaging.
    # logging.info(f"{context} - Raw aggregated data:\n{df.describe()}")
    
    # We'll check each date row for outliers.
    median_abs = df.abs().median().median()  # median of medians as a rough baseline
    threshold = 10 * median_abs if median_abs > 0 else 1e6
    for date_idx in df.index:
        row_vals = df.loc[date_idx]
        outliers = row_vals[abs(row_vals) > threshold]
        if not outliers.empty:
            logging.warning(f"{context} - Potential outliers on {date_idx} with threshold {threshold:.2f}:\n{outliers}")
    
    # Compute mean along columns, ignoring NaNs.
    return df.mean(axis=1, skipna=True)

def main():
    logging.info("Starting sector trend visualization process.")
    
    # Step 1: Extract data from Excel files.
    extracted_data = traverse_and_process(ROOT_DIR)
    logging.info(f"Extracted data for {len(extracted_data)} files.")
    
    # Organize companies by sector.
    # Structure: { sector: { company: { ratio: trend_series, ... } } }
    sector_company_trends = {}
    for file_path, sheets in extracted_data.items():
        company_name = os.path.splitext(os.path.basename(file_path))[0]
        sector = os.path.basename(os.path.dirname(file_path))
        # Get ratio trend series for the company.
        ratio_trends = compute_ratios_for_company(sheets)
        if not ratio_trends:
            logging.warning(f"No ratio trends for company {company_name} in sector {sector}, skipping.")
            continue
        if sector not in sector_company_trends:
            sector_company_trends[sector] = {}
        sector_company_trends[sector][company_name] = ratio_trends
    
    # For each sector, aggregate the trend series for each ratio.
    output_dir = "output_sector_trends"
    os.makedirs(output_dir, exist_ok=True)
    
    # We'll iterate over each sector.
    for sector, companies in sector_company_trends.items():
        logging.info(f"Processing sector: {sector}")
        # Prepare a dictionary to collect lists of series per ratio.
        # Keys are ratio names (e.g., "Brüt Kar/Hasılat") and values are lists of series.
        ratio_series_dict = {ratio: [] for ratio in [
            "Brüt Kar/Hasılat", "Yönetim/Kar", "Pazarlama/Kar", "Arge/Kar",
            "Amortisman/Kar", "Dönem Net/Brüt Kar", "Dönem Net/Toplam Varlık",
            "Finansal Borç/Özkaynak", "Sermaye Harcaması/Dönem Net",
            "Geçmiş Yıl Kar + Yedek Akçe", "Hisse Başı Kazanç", "Hisse Başı Sermaye Harcaması"
        ]}
        # Loop over companies and add available trend series.
        for company, trends in companies.items():
            for ratio, series in trends.items():
                # Only add non-empty series.
                if isinstance(series, pd.Series) and not series.empty:
                    ratio_series_dict[ratio].append(series)
        
        # Compute the average trend for each ratio.
        avg_trends = {}
        # Updated code:
        for ratio, series_list in ratio_series_dict.items():
            context_str = f"Sector {sector} - Ratio {ratio}"
            avg_series = aggregate_trend_series(series_list, context=context_str)
            avg_trends[ratio] = avg_series
        
        # Create a figure with subplots arranged in a grid (3 columns, 4 rows for 12 ratios)
        n_ratios = len(avg_trends)
        cols = 3
        rows = math.ceil(n_ratios / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(18, 6 * rows), squeeze=False)
        axs = axs.flatten()
        
        # Plot each ratio's average trend
        for idx, (ratio, series) in enumerate(avg_trends.items()):
            ax = axs[idx]
            if series.empty:
                ax.text(0.5, 0.5, f"No data for {ratio}", horizontalalignment='center', verticalalignment='center')
                ax.set_title(ratio)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.plot(series.index, series.values, marker='o', linestyle='-')
            ax.set_title(ratio)
            ax.set_xlabel("Date")
            ax.set_ylabel("Average Value")
            ax.grid(True)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        # Hide unused subplots, if any.
        for extra_ax in axs[n_ratios:]:
            extra_ax.set_visible(False)
        
        fig.suptitle(f"Sector '{sector}' - Average Ratio Trends", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(output_dir, f"{sector}_average_trends.png")
        fig.savefig(output_path)
        logging.info(f"Saved sector trend visualization for '{sector}' to {output_path}")
        plt.close(fig)
    
    logging.info("Sector trend visualization process completed.")

if __name__ == "__main__":
    main()