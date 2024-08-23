import pandas as pd

# Read the CSV files
df1 = pd.read_csv('filtered_peptides_label_1.csv')
df2 = pd.read_csv('filtered_peptides_label_0.csv')

# Concatenate the dataframes
combined_df = pd.concat([df1, df2])

# Write the combined dataframe to a new CSV file
combined_df.to_csv('combined.csv', index=False)