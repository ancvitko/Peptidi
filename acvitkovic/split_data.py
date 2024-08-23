import pandas as pd

# Read the dataset
df = pd.read_csv('peptides_data.csv')

# Split the dataset based on the label
df_label_0 = df[df['label'] == 0]
df_label_1 = df[df['label'] == 1]

# Write the datasets to new CSV files
df_label_0.to_csv('peptides_label_0.csv', index=False)
df_label_1.to_csv('peptides_label_1.csv', index=False)

print("Files created successfully.")
