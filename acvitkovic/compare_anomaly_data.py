import pandas as pd

# Step 1: Load the CSV files into DataFrames
anomaly_data = pd.read_csv('anomaly_data.csv')
combined_data = pd.read_csv('combined.csv')

# Step 2: Ensure that the 'index' column in anomaly_data is set as the index if it's not already
anomaly_data.set_index('index', inplace=True)

# Step 3: Reset the index of combined_data to match the indexing of anomaly_data
combined_data.reset_index(inplace=True)

# Step 4: Merge the DataFrames on the 'index' column to align the anomaly labels
merged_data = pd.merge(combined_data, anomaly_data, left_index=True, right_index=True, suffixes=('', '_anomaly_data'))

# Step 5: Calculate the accuracy
merged_data['correct'] = merged_data['anomaly'] == merged_data['anomaly_anomaly_data']
accuracy = merged_data['correct'].mean()

print(f'Accuracy: {accuracy * 100:.2f}%')
