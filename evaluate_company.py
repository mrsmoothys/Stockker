#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from gpt_all_in_one import traverse_and_process, compute_ratios_for_company, ROOT_DIR
from scipy.stats import linregress
import logging

# Remove any existing logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('evaluation.log', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Define the list of ratios with their types ("profit" or "cost")
ratio_types = {
    "Brüt Kar/Hasılat": "profit",
    "Yönetim/Kar": "cost",
    "Pazarlama/Kar": "cost",
    "Arge/Kar": "cost",
    "Amortisman/Kar": "cost",
    "Dönem Net/Brüt Kar": "profit",
    "Dönem Net/Toplam Varlık": "profit",
    "Finansal Borç/Özkaynak": "cost",  # lower is better
    "Sermaye Harcaması/Dönem Net": "cost",
    "Geçmiş Yıl Kar + Yedek Akçe": "profit",
    "Hisse Başı Kazanç": "profit",
    "Hisse Başı Sermaye Harcaması": "cost"
}

# Settings for scoring
MAX_PERF_SCORE = 5.0   # Maximum performance score per ratio
MAX_TREND_SCORE = 5.0  # Maximum trend consistency score per ratio
IPO_THRESHOLD = 6      # If a ratio's trend series has fewer than 6 data points, flag as IPO

def compute_trend_metrics(series):
    """
    Given a trend series (pandas Series with datetime index),
    compute:
      - average (mean)
      - standard deviation (std)
      - slope (using linear regression on date ordinal values)
    Returns a tuple: (avg, std, slope, n_points)
    """
    if series.empty:
        return (np.nan, np.nan, np.nan, 0)
    # Ensure the series is sorted by date
    series = series.sort_index()
    n_points = len(series)
    avg = series.mean()
    std = series.std()
    # Convert datetime index to ordinal numbers for regression
    x = np.array([dt.toordinal() for dt in series.index])
    y = series.values
    # If there are fewer than 2 points, slope is undefined
    if len(x) < 2:
        slope = 0
    else:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return (avg, std, slope, n_points)

def performance_score(company_avg, sector_avg, ratio_type):
    """
    Compute the performance score for one ratio.
    For profit ratios: if company_avg > sector_avg, score = MAX_PERF_SCORE * min(1, (company_avg/sector_avg)-1), else 0.
    For cost ratios: if company_avg < sector_avg, score = MAX_PERF_SCORE * min(1, 1 - (company_avg/sector_avg)), else 0.
    """
    if sector_avg == 0 or np.isnan(company_avg) or np.isnan(sector_avg):
        return 0
    if ratio_type == "profit":
        if company_avg > sector_avg:
            diff = (company_avg / sector_avg) - 1
            return MAX_PERF_SCORE * min(1, diff)
        else:
            return 0
    else:  # cost ratio
        if company_avg < sector_avg:
            diff = 1 - (company_avg / sector_avg)
            return MAX_PERF_SCORE * min(1, diff)
        else:
            return 0

def trend_consistency_score(avg, std, slope, ratio_type):
    """
    Compute the trend consistency score for a ratio.
    - Stability: 5 * (1 - min(1, std/avg)) if avg > 0, else 0.
    - Direction: For profit ratios, if slope > 0, score = 5 * min(1, slope/avg); for cost ratios, if slope < 0, score = 5 * min(1, -slope/avg); else 0.
    Then average the two components.
    """
    if np.isnan(avg) or avg == 0:
        return 0
    # Stability: lower std relative to avg is better
    stability = MAX_TREND_SCORE * (1 - min(1, std / avg))
    # Direction: favorable slope yields points
    if ratio_type == "profit":
        direction = MAX_TREND_SCORE * min(1, slope / avg) if slope > 0 else 0
    else:
        direction = MAX_TREND_SCORE * min(1, -slope / avg) if slope < 0 else 0
    return (stability + direction) / 2.0

def evaluate_company_ratios(company_ratios):
    """
    Given a dictionary of company ratio metrics, where each key is a ratio name and the value is a dict:
      { 'avg': ..., 'std': ..., 'slope': ..., 'n': ... }
    Returns a dictionary with:
      'performance': total performance score (sum over ratios),
      'consistency': total trend consistency score (sum over ratios),
      'total': sum of the two,
      and also a dictionary of individual ratio scores.
      Also include an IPO flag if for any ratio n < IPO_THRESHOLD.
    """
    total_perf = 0
    total_cons = 0
    ratio_scores = {}
    ipo_flag = False
    for ratio, metrics in company_ratios.items():
        avg = metrics.get("avg", np.nan)
        std = metrics.get("std", np.nan)
        slope = metrics.get("slope", 0)
        n = metrics.get("n", 0)
        # Mark as IPO if very few data points
        if n < IPO_THRESHOLD:
            ipo_flag = True
        # We'll compute performance later when we know sector averages.
        # For now, store avg, std, slope.
        ratio_scores[ratio] = {"avg": avg, "std": std, "slope": slope, "n": n}
    # Return the raw metrics dictionary; the performance part will be computed once we have sector averages.
    return ratio_scores, ipo_flag

def main():
    logging.info('Starting evaluation process.')
    # Step 1: Extract data and compute ratio trends for each company.
    extracted_data = traverse_and_process(ROOT_DIR)
    logging.info(f'Extracted data for {len(extracted_data)} files.')
    # Data structure: { sector: { company: { ratio: {avg, std, slope, n} } } }
    sector_company_metrics = {}
    
    for file_path, sheets in extracted_data.items():
        company_name = os.path.splitext(os.path.basename(file_path))[0]
        sector = os.path.basename(os.path.dirname(file_path))
        ratios_trend = compute_ratios_for_company(sheets)
        company_metrics = {}
        for ratio, series in ratios_trend.items():
            avg, std, slope, n = compute_trend_metrics(series)
            company_metrics[ratio] = {"avg": avg, "std": std, "slope": slope, "n": n}
        if not company_metrics:
            logging.warning(f'No valid metrics for company: {company_name} in sector: {sector}, skipping.')
            continue
        if sector not in sector_company_metrics:
            sector_company_metrics[sector] = {}
        sector_company_metrics[sector][company_name] = company_metrics
    
    # Step 2: For each sector, compute sector averages for each ratio (from company averages).
    sector_results = {}
    for sector, companies in sector_company_metrics.items():
        logging.info(f'Processing sector: {sector}')
        # DataFrame for each ratio's average values
        # Updated code:
        df_avg = pd.DataFrame({company: {ratio: metrics["avg"] for ratio, metrics in comp_metrics.items()} 
                            for company, comp_metrics in companies.items()}).T
        # Replace Inf and -Inf with NaN so they do not affect the average calculation
        df_avg = df_avg.replace([np.inf, -np.inf], np.nan)
        sector_avg = df_avg.mean(skipna=True)
        
        # Now, evaluate each company
        company_scores = {}
        for company, metrics in companies.items():
            logging.info(f'Evaluating company: {company} in sector: {sector}')
            perf_score = 0
            cons_score = 0
            ratio_detail = {}
            for ratio, m in metrics.items():
                if np.isnan(m["avg"]) or np.isinf(m["avg"]):
                    logging.warning(f"Invalid average for ratio {ratio} for company {company} in sector {sector}: avg={m['avg']}. Omitting from scoring.")
                    continue
                # Performance score relative to sector average
                sec_avg = sector_avg.get(ratio, np.nan)
                if np.isnan(sec_avg) or sec_avg == 0:
                    perf = 0
                else:
                    if ratio_types.get(ratio, "profit") == "profit":
                        if m["avg"] > sec_avg:
                            perf = MAX_PERF_SCORE * min(1, (m["avg"] / sec_avg) - 1)
                        else:
                            perf = 0
                    else:  # cost ratio
                        if m["avg"] < sec_avg:
                            perf = MAX_PERF_SCORE * min(1, 1 - (m["avg"] / sec_avg))
                        else:
                            perf = 0
                # Trend consistency score
                cons = trend_consistency_score(m["avg"], m["std"], m["slope"], ratio_types.get(ratio, "profit"))
                ratio_detail[ratio] = {"avg": m["avg"], "perf": perf, "cons": cons, "n": m["n"]}
                perf_score += perf
                cons_score += cons
            total = perf_score + cons_score
            company_scores[company] = {
                "Performance Score": perf_score,
                "Trend Consistency Score": cons_score,
                "Total Score": total,
                "Details": ratio_detail
            }
        sector_results[sector] = {"sector_avg": sector_avg, "companies": company_scores}
    
    # Step 3: Output results for each sector
    output_dir = "output_evaluations"
    os.makedirs(output_dir, exist_ok=True)
    
    for sector, data in sector_results.items():
        print(f"\nSector: {sector}")
        print("Sector Averages for Ratios:")
        print(data["sector_avg"])
        companies = data["companies"]
        df_scores = pd.DataFrame({comp: {"Performance Score": scores["Performance Score"],
                                         "Trend Consistency Score": scores["Trend Consistency Score"],
                                         "Total Score": scores["Total Score"]}
                                  for comp, scores in companies.items()}).T
        df_scores = df_scores.sort_values(by="Total Score", ascending=False)
        print("\nTop Companies by Total Score:")
        print(df_scores.head(5))
        # Save the detailed scores per sector
        df_scores.to_csv(os.path.join(output_dir, f"{sector}_evaluation_scores.csv"))
    
    # Optionally, consolidate across sectors:
    consolidated = {}
    for sector, data in sector_results.items():
        for company, scores in data["companies"].items():
            consolidated[f"{sector}/{company}"] = scores["Total Score"]
    consolidated_series = pd.Series(consolidated).sort_values(ascending=False)
    print("\nConsolidated Evaluation Scores (all sectors):")
    print(consolidated_series.head(10))
    consolidated_series.to_csv(os.path.join(output_dir, "consolidated_evaluation_scores.csv"))
    
if __name__ == "__main__":
    main()