import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv('encoded_properties.csv')

# Split Data into Training and Test Sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train Isolation Forest on Training Data
iso_forest = IsolationForest(contamination='auto', random_state=42)
iso_forest.fit(train_data)

# Add anomaly scores to the Training DataFrame
train_data['anomaly_score'] = iso_forest.decision_function(train_data)

# Identify Outliers in the Training Data
train_data['anomaly'] = iso_forest.predict(train_data.drop(columns=['anomaly_score']))

# Sort training data by anomaly score in ascending order (worst scores first)
train_data_sorted = train_data.sort_values(by='anomaly_score', ascending=True)

# Remove Outliers from Training Data
outlier_fraction = 0.1  # Define the proportion of outliers to remove
num_outliers = int(len(train_data_sorted) * outlier_fraction)
train_data_filtered = train_data_sorted.iloc[num_outliers:]

# Save Filtered Training Data
train_data_filtered.to_csv('filtered_train_peptides.csv', index=False)
test_data.to_csv('test_peptides.csv', index=False)
