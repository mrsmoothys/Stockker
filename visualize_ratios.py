#!/usr/bin/env python3
import os
import math
import matplotlib.pyplot as plt
from gpt_all_in_one import traverse_and_process, compute_ratios_for_company, ROOT_DIR

def main():
    # Extract data from Excel files
    extracted_data = traverse_and_process(ROOT_DIR)
    
    results = {}
    # Compute ratios for each company
    for file_path, sheets in extracted_data.items():
        # Derive company name from file name (without extension)
        company_name = os.path.splitext(os.path.basename(file_path))[0]
        ratios = compute_ratios_for_company(sheets)
        results[company_name] = ratios
    
    # Create an output directory for the saved images
    output_dir = "output_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    # For each company, create one figure with subplots for every non-empty ratio trend
    for company, ratios in results.items():
        # Filter out empty series
        valid_ratios = {name: series for name, series in ratios.items() if not series.empty}
        if not valid_ratios:
            print(f"{company} has no valid ratio trends, skipping visualization.")
            continue
        
        n_subplots = len(valid_ratios)
        # We'll arrange subplots in 3 columns, computing the number of rows needed.
        cols = 3
        rows = math.ceil(n_subplots / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(18, 6 * rows), squeeze=False)
        axs = axs.flatten()  # Flatten so we can iterate in a single loop
        
        for idx, (ratio_name, series) in enumerate(valid_ratios.items()):
            ax = axs[idx]
            ax.plot(series.index, series.values, marker='o', linestyle='-')
            ax.set_title(ratio_name)
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.grid(True)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        
        # Hide any unused subplots
        for extra_ax in axs[n_subplots:]:
            extra_ax.set_visible(False)
        
        fig.suptitle(f"Financial Ratio Trends for {company}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(output_dir, f"{company}.png")
        fig.savefig(output_path)
        print(f"Saved visualization for {company} to {output_path}")
        plt.close(fig)

if __name__ == "__main__":
    main()