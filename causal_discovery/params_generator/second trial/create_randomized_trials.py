import pandas as pd

# === Parameters ===
input_csv = '/home/forough/Desktop/causal_navigation/src/causal_discovery/params_generator/parameters_updated.csv'           # Path to your original CSV file
output_csv = 'Nav2_random_configurations2.csv' # Path to save the new CSV file
num_rows_to_select = 200          # Number of random rows to select

# === Load CSV ===
df = pd.read_csv(input_csv)
df['original_row_index'] = df.index + 2
# === Sample Rows ===
sampled_df = df.sample(n=num_rows_to_select, random_state=50)  # Use random_state for reproducibility
cols = ['original_row_index'] + [col for col in sampled_df.columns if col != 'original_row_index']
sampled_df = sampled_df[cols]
# === Save to New CSV ===

sampled_df.to_csv(output_csv, index=False)

print(f"Saved {num_rows_to_select} randomly selected rows to {output_csv}")
