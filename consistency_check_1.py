import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Load your dataset
data = pd.read_csv('encoded_properties.csv')
X = data.drop(columns=['sequence', 'label'], errors='ignore')

# Specify the number of iterations
n_iterations = 10

# Store anomaly results across iterations
anomalies_dict = {}

for i in range(n_iterations):
    # Train Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=i)
    clf.fit(X)
    
    # Get anomaly labels and decision scores
    anomaly_labels = clf.predict(X)
    decision_scores = clf.decision_function(X)
    
    # Add results to the DataFrame
    data[f'anomaly_label_{i}'] = anomaly_labels
    data[f'decision_score_{i}'] = decision_scores

    # Record indices of anomalies
    for idx in data[data[f'anomaly_label_{i}'] == -1].index:
        if idx in anomalies_dict:
            anomalies_dict[idx].append(data[f'decision_score_{i}'].iloc[idx])
        else:
            anomalies_dict[idx] = [data[f'decision_score_{i}'].iloc[idx]]

# Find consistent anomalies
consistent_anomalies = {k: v for k, v in anomalies_dict.items() if len(v) == n_iterations}

# Sort consistent anomalies by the average decision score
sorted_anomalies = sorted(consistent_anomalies.items(), key=lambda item: np.mean(item[1]))

# Get the top 10 consistent anomalies
top_10_anomalies = sorted_anomalies[:10]

# Print out the information for the top 10 consistent anomalies
for idx, scores in top_10_anomalies:
    print(f"Row index: {idx}")
    print(f"Original data: {data.loc[idx, data.columns[:19]]}")  # Assuming the first 19 columns are your original data
    print(f"Anomaly decision scores: {sorted(scores)}")
    print("\n")

# Create a DataFrame to store the summary of the top 10 consistent anomalies
summary_df = pd.DataFrame(columns=['index', 'original_data', 'anomaly_decision_scores'])

for idx, scores in top_10_anomalies:
    summary_df = summary_df.append({
        'index': idx,
        'original_data': data.loc[idx, data.columns[:19]].to_dict(),
        'anomaly_decision_scores': sorted(scores)
    }, ignore_index=True)

# Print the summary DataFrame
print(summary_df)
