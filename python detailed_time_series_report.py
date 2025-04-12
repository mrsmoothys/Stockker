#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import logging
import json  # Ensure this is imported at the top if not already
from gpt_all_in_one import traverse_and_process, compute_ratios_for_company, ROOT_DIR

# Set up logging (both file and console)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('detailed_time_series_report.log', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# List of ratios to report
ratios_list = [
    "Brüt Kar/Hasılat", 
    "Yönetim/Kar", 
    "Pazarlama/Kar", 
    "Arge/Kar",
    "Amortisman/Kar", 
    "Dönem Net/Brüt Kar", 
    "Dönem Net/Toplam Varlık",
    "Finansal Borç/Özkaynak", 
    "Sermaye Harcaması/Dönem Net", 
    "Geçmiş Yıl Kar + Yedek Akçe", 
    "Hisse Başı Kazanç", 
    "Hisse Başı Sermaye Harcaması"
]

ratio_types = {
    "Brüt Kar/Hasılat": "profit",
    "Yönetim/Kar": "cost",
    "Pazarlama/Kar": "cost",
    "Arge/Kar": "cost",
    "Amortisman/Kar": "cost",
    "Dönem Net/Brüt Kar": "profit",
    "Dönem Net/Toplam Varlık": "profit",
    "Finansal Borç/Özkaynak": "cost",
    "Sermaye Harcaması/Dönem Net": "cost",
    "Geçmiş Yıl Kar + Yedek Akçe": "profit",
    "Hisse Başı Kazanç": "profit",
    "Hisse Başı Sermaye Harcaması": "cost"
}


def format_ratio_details(details):
    """
    Given a details dictionary for a company (with keys as ratio names and values containing
    avg, perf, cons, and n), return a nicely formatted multiline string.
    """
    lines = []
    for ratio, values in details.items():
        lines.append(f"{ratio}:")
        lines.append(f"  avg: {values['avg']:.3f}")
        lines.append(f"  perf: {values['perf']:.3f}")
        lines.append(f"  cons: {values['cons']:.3f}")
        lines.append(f"  n: {values['n']}")
    return "\n".join(lines)

# In the generate_evaluation_excel_report function, replace the line that writes the Details column:
# Original line:
# "Details": json.dumps(scores["Details"], indent=2)
# Replace it with:
# "Details": format_ratio_details(scores["Details"])

def compute_trend_metrics(series):
    """
    Given a pandas Series with a datetime index, compute trend metrics:
      - average (mean) of the series,
      - standard deviation,
      - slope (via linear regression using ordinal dates),
      - number of data points.
    Returns a tuple: (avg, std, slope, n).
    """
    if series.empty:
        return (np.nan, np.nan, 0, 0)
    avg = series.mean()
    std = series.std()
    n = len(series)
    # Convert datetime index to numeric values using ordinal
    x = np.array([dt.toordinal() for dt in series.index])
    y = series.values
    if n < 2:
        slope = 0
    else:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return (avg, std, slope, n)


def gather_time_series_details():
    """
    Use the existing extraction and ratio calculation functions to gather raw trend series for each company.
    Returns a dictionary: 
      { sector: { company: { ratio: trend_series (pd.Series), ... } } }
    """
    extracted_data = traverse_and_process(ROOT_DIR)
    sector_company_details = {}
    for file_path, sheets in extracted_data.items():
        company_name = os.path.splitext(os.path.basename(file_path))[0]
        sector = os.path.basename(os.path.dirname(file_path))
        ratio_trends = compute_ratios_for_company(sheets)
        if not ratio_trends:
            logging.warning(f"No ratio trends for company {company_name} in sector {sector}, skipping.")
            continue
        if sector not in sector_company_details:
            sector_company_details[sector] = {}
        sector_company_details[sector][company_name] = ratio_trends
    return sector_company_details

def create_ratio_table_for_sector(sector, companies_data, ratio_name):
    """
    For a given sector and ratio name, build a DataFrame where:
      - The index is the union of all dates available for that ratio across companies.
      - Each column corresponds to one company’s trend series for that ratio.
      - A final column "Sector Average" is computed as the row-wise robust average (median).
    Returns the DataFrame.
    """
    series_list = {}
    for company, ratios in companies_data.items():
        if ratio_name in ratios:
            series = ratios[ratio_name]
            if isinstance(series, pd.Series) and not series.empty:
                series_list[company] = series
    if not series_list:
        return pd.DataFrame()  # no data available
    
    # Align all series on the union of their dates.
    df = pd.concat(series_list, axis=1, join='outer')
    df.sort_index(inplace=True)
    
    # Robust outlier handling:
    # 1. Replace Inf and -Inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 2. Winsorize: For each column, clip values at the 1st and 99th percentiles
    for col in df.columns:
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # 3. Compute sector average for each date using the median for robustness.
    df["Sector Average"] = df.median(axis=1, skipna=True)
    return df

def generate_time_series_excel_report(details, output_file="detailed_time_series_report.xlsx"):
    """
    Generate an Excel report with one worksheet per sector.
    In each worksheet, for each ratio (from ratios_list), a table is written showing:
      - Dates (from trend series)
      - Each company's value for that ratio on that date
      - A column for the computed "Sector Average"
    A chart is inserted for each ratio that visualizes every company's data against the sector average.
    """
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        # Process each sector
        for sector, companies in details.items():
            sheet_name = sector[:31]  # Excel sheet name limit is 31 characters.
            logging.info(f"Processing sector: {sector}")
            # Start writing from row 0
            current_row = 0
            # Write a header for the sector.
            header_df = pd.DataFrame({"Sector": [sector]})
            header_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=current_row)
            current_row += 2  # Leave one blank row
            
            # For each ratio, if data exists, create a table and chart.
            for ratio in ratios_list:
                df_ratio = create_ratio_table_for_sector(sector, companies, ratio)
                if df_ratio.empty:
                    logging.info(f"No data for ratio {ratio} in sector {sector}.")
                    continue
                worksheet = writer.sheets[sheet_name]
                # Write the ratio name as a title
                worksheet.write(current_row, 0, ratio)
                current_row += 1
                # Write the table starting at the current row.
                table_start = current_row
                df_ratio.reset_index(inplace=True)  # Move the index (dates) to a column.
                df_ratio.rename(columns={'index': 'Date'}, inplace=True)
                df_ratio.to_excel(writer, sheet_name=sheet_name, index=False, startrow=current_row)
                num_table_rows = len(df_ratio) + 1  # header row plus data rows.
                table_end = current_row + num_table_rows - 1
                current_row += num_table_rows + 1  # Leave a blank row after the table.
                
                # Create a chart for this ratio.
                chart = workbook.add_chart({'type': 'line'})
                n_cols = df_ratio.shape[1]  # Total number of columns (Date, Company1, Company2, ..., Sector Average)
                # Add series for each column except the Date column.
                for col in range(1, n_cols):
                    chart.add_series({
                        'name':       [sheet_name, table_start, col],
                        'categories': [sheet_name, table_start + 1, 0, table_end, 0],
                        'values':     [sheet_name, table_start + 1, col, table_end, col],
                    })
                chart.set_title({'name': ratio})
                chart.set_x_axis({'name': 'Date'})
                chart.set_y_axis({'name': 'Value'})
                chart.set_style(10)
                # Insert the chart a few rows below the table.
                chart_row = table_end + 2
                worksheet.insert_chart(chart_row, 0, chart)
                current_row = chart_row + 15  # Update current_row after chart insertion.
            logging.info(f"Completed writing sector {sector} to Excel.")
        logging.info(f"Excel report generated: {output_file}")

# --- New Code Added at the Bottom of the File ---

def compute_performance_score(company_avg, sector_avg, ratio_type):
    """
    Compute the performance score for a single ratio.
    For profit ratios, if company_avg > sector_avg, score = 5 * min(1, (company_avg/sector_avg - 1)).
    For cost ratios, if company_avg < sector_avg, score = 5 * min(1, 1 - (company_avg/sector_avg)).
    """
    if sector_avg == 0 or np.isnan(company_avg) or np.isnan(sector_avg):
        return 0
    if ratio_type == "profit":
        if company_avg > sector_avg:
            diff = (company_avg / sector_avg) - 1
            return 5 * min(1, diff)
        else:
            return 0
    else:  # cost ratio
        if company_avg < sector_avg:
            diff = 1 - (company_avg / sector_avg)
            return 5 * min(1, diff)
        else:
            return 0

def trend_consistency_score(avg, std, slope, ratio_type):
    """
    Compute the trend consistency score for a ratio.
    Uses two components:
      - Stability: lower std relative to |avg| is better.
      - Direction: For profit ratios, a positive slope is good; for cost ratios, a negative slope is good.
    The score is the average of these two components (maximum 5 points per ratio).
    """
    if np.isnan(avg) or abs(avg) < 1e-6:
        return 0
    # Stability component: lower standard deviation relative to the absolute average is better.
    stability = 5 * (1 - min(1, std / abs(avg)))
    # Direction component: for profit ratios, a positive slope is beneficial; for cost ratios, a negative slope is beneficial.
    if ratio_type == "profit":
        direction = 5 * min(1, slope / abs(avg)) if slope > 0 else 0
    else:  # cost ratio
        direction = 5 * min(1, -slope / abs(avg)) if slope < 0 else 0
    return (stability + direction) / 2.0

def compute_company_evaluation(details):
    """
    Compute an overall evaluation score for each company.
    For each ratio, the score is the sum of:
      - Performance Score (based on company average vs. sector average)
      - Trend Consistency Score (based on trend metrics: avg, std, and slope)
    Returns a dictionary: { sector: { company: { "Total Score": score, "Details": { ratio: { ... } } } } }
    """
    evaluation = {}
    for sector, companies in details.items():
        # Build a DataFrame for each ratio's average values per company.
        data = {}
        # Also, store the detailed trend metrics per ratio for each company.
        metrics_data = {}
        for company, ratios in companies.items():
            company_data = {}
            company_metrics = {}
            for ratio in ratios_list:
                if ratio in ratios and isinstance(ratios[ratio], pd.Series) and not ratios[ratio].empty:
                    # Compute trend metrics: average, standard deviation, slope, number of data points.
                    avg, std, slope, n = compute_trend_metrics(ratios[ratio])
                    company_data[ratio] = avg
                    company_metrics[ratio] = {"avg": avg, "std": std, "slope": slope, "n": n}
                else:
                    company_data[ratio] = np.nan
                    company_metrics[ratio] = {"avg": np.nan, "std": np.nan, "slope": 0, "n": 0}
            data[company] = company_data
            metrics_data[company] = company_metrics
        if not data:
            continue
        df = pd.DataFrame.from_dict(data, orient="index")
        # Replace Inf and -Inf with NaN.
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Compute sector average for each ratio (using median for robustness).
        sector_avg = df.median(skipna=True)
        company_scores = {}
        # Now, compute scores for each company.
        for company in df.index:
            total_score = 0
            ratio_detail = {}
            for ratio in ratios_list:
                comp_val = df.at[company, ratio]
                sec_val = sector_avg.get(ratio, np.nan)
                # Compute performance score.
                if np.isnan(comp_val) or np.isnan(sec_val):
                    perf = 0
                else:
                    perf = compute_performance_score(comp_val, sec_val, ratio_types.get(ratio, "profit"))
                # Compute trend consistency score using the company's computed metrics.
                m = metrics_data[company][ratio]
                cons = trend_consistency_score(m["avg"], m["std"], m["slope"], ratio_types.get(ratio, "profit"))
                ratio_detail[ratio] = {"avg": m["avg"], "perf": perf, "cons": cons, "n": m["n"]}
                total_score += (perf + cons)
            company_scores[company] = {"Total Score": max(total_score, 0), "Details": ratio_detail}
        evaluation[sector] = company_scores
    return evaluation

def generate_evaluation_excel_report(evaluation_data, output_file="evaluation_report.xlsx"):
    """
    Generate an Excel file summarizing the evaluation scores for each company in each sector.
    The report includes columns for Company, Total Score, and a nicely formatted string with detailed ratio scores.
    """
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for sector, companies in evaluation_data.items():
            # Prepare data for the report.
            rows = []
            for company, scores in companies.items():
                rows.append({
                    "Company": company,
                    "Total Score": scores["Total Score"],
                    "Details": format_ratio_details(scores["Details"])
                })
            df_sector = pd.DataFrame(rows)
            df_sector.sort_values(by="Total Score", ascending=False, inplace=True)
            sheet_name = sector[:31]
            df_sector.to_excel(writer, sheet_name=sheet_name, index=False)
    logging.info(f"Evaluation Excel report generated: {output_file}")

def generate_evaluation_png(evaluation_data, output_dir="evaluation_pngs"):
    """
    Generate PNG bar charts for each sector showing company evaluation scores.
    """
    os.makedirs(output_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    for sector, companies in evaluation_data.items():
        df_sector = pd.DataFrame(list(companies.items()), columns=["Company", "Total Score"])
        df_sector.sort_values(by="Total Score", ascending=False, inplace=True)
        plt.figure(figsize=(10, 6))
        plt.barh(df_sector["Company"], df_sector["Total Score"], color='skyblue')
        plt.xlabel("Total Evaluation Score")
        plt.title(f"Evaluation Scores for Sector: {sector}")
        plt.gca().invert_yaxis()  # Highest score on top
        plt.tight_layout()
        png_file = os.path.join(output_dir, f"{sector}_evaluation.png")
        plt.savefig(png_file)
        plt.close()
        logging.info(f"Saved evaluation chart for sector {sector} to {png_file}")

def run_best_company_picker(details):
    """
    Compute evaluation scores using compute_company_evaluation,
    generate an Excel report and PNG charts.
    """
    evaluation_data = compute_company_evaluation(details)
    generate_evaluation_excel_report(evaluation_data)
    generate_evaluation_png(evaluation_data)
    # For each sector, log top 5 companies.
    for sector, companies in evaluation_data.items():
        sorted_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_companies[:5]
        logging.info(f"Top 5 companies in sector {sector}: {top_5}")

def main():
    logging.info("Starting detailed time series report generation.")
    details = gather_time_series_details()
    generate_time_series_excel_report(details)
    logging.info("Detailed time series report generation completed.")
    
    # Run the best company picker algorithm.
    logging.info("Starting company evaluation and best company picking.")
    run_best_company_picker(details)
    logging.info("Company evaluation and best company picking completed.")

if __name__ == "__main__":
    main()