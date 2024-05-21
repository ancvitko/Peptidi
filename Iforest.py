import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the CSV file into a DataFrame
df = pd.read_csv('encoded_properties.csv')

# Check for missing values and fill them if necessary
df.fillna(df.mean(), inplace=True)

# Features for Isolation Forest (excluding any labels if present)
features = df.drop(columns=['sequence', 'label'], errors='ignore')

# Initialize the Isolation Forest model
iso_forest = IsolationForest(bootstrap=True, max_samples=2048 ,contamination=0.394, random_state=2546)

# Fit the model on the data
iso_forest.fit(features)

# Predict anomalies
anomalies = iso_forest.predict(features)

# -1 indicates an anomaly, 1 indicates a normal point
df['anomaly'] = anomalies

# Save the DataFrame with anomaly column to a CSV file
df.to_csv('encoded_properties_with_anomalies.csv', index=False)

# Filter the DataFrame to show only the anomalies
anomalies_df = df[df['anomaly'] == -1]

# Save the anomalies to a new CSV file
anomalies_df.to_csv('anomalies.csv', index=False)# Save the DataFrame with anomaly column to a CSV file
df.to_csv('encoded_properties_with_anomalies.csv', index=False)

# Filter the DataFrame to show only the anomalies
anomalies_df = df[df['anomaly'] == -1]

# Save the anomalies to a new CSV file
anomalies_df.to_csv('anomalies.csv', index=False)
