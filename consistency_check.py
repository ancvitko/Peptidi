import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

data = pd.read_csv('encoded_properties.csv')
X = data.drop(columns=['sequence', 'label'], errors='ignore')

# Function to run Isolation Forest and get anomaly indices
def get_anomalies_with_scores(X, random_state, cont, max_samples):
    clf = IsolationForest(contamination=cont, random_state=random_state, max_samples = max_samples)
    clf.fit(X)
    # Predict anomalies (-1 for anomaly, 1 for normal)
    anomaly_labels = clf.predict(X)
    anomaly_indices = np.where(anomaly_labels == -1)[0]
    decision_scores = clf.decision_function(X)
    return anomaly_indices, decision_scores

# Run Isolation Forest multiple times and track anomalies
num_runs = 10
all_anomalies = []
all_scores = []

for i in range(num_runs):
    anomalies, scores = get_anomalies_with_scores(X, random_state=25, cont= 0.19, max_samples=256*(i+2))
    all_anomalies.append(anomalies)
    all_scores.append(scores)

anomaly_counts = pd.DataFrame(0, index=X.index, columns=['count'])
decision_scores_df = pd.DataFrame(0, index=X.index, columns=[f'run_{i}' for i in range(num_runs)])

# Count the number of times each index is detected as an anomaly
# and store the decision scores
for run_index, (anomalies, scores) in enumerate(zip(all_anomalies, all_scores)):
    anomaly_counts.loc[anomalies, 'count'] += 1
    decision_scores_df[f'run_{run_index}'] = scores

# Add the average decision score to the main DataFrame
data['average_decision_score'] = decision_scores_df.mean(axis=1)


print(anomaly_counts)

#Filter rows that are consistently marked as anomalies
consistent_anomalies = anomaly_counts[anomaly_counts['count'] == num_runs]
print('Consistent Anomalies:')
print(consistent_anomalies)
consistent_anomaly_indices = consistent_anomalies.index
sorted_consistent_anomalies = data.loc[consistent_anomaly_indices].sort_values(by='average_decision_score')


print('Consistent Anomalies with Sorted Decision Scores:')
print(sorted_consistent_anomalies[['average_decision_score']])


sorted_consistent_anomaly_indices = sorted_consistent_anomalies.index
sorted_consistent_anomaly_scores = decision_scores_df.loc[sorted_consistent_anomaly_indices]


print('Decision Scores of Consistent Anomalies for Each Run (Sorted):')
print(sorted_consistent_anomaly_scores)


#Filter rows that are inconsistently marked as anomalies
inconsistent_anomalies = anomaly_counts[ np.logical_and(anomaly_counts['count'] != num_runs, anomaly_counts['count'] > 0 )]
print('Inconsistent Anomalies:')

inconsistent_anomaly_indices = inconsistent_anomalies.index
print(inconsistent_anomalies)

sorted_inconsistent_anomalies = data.loc[inconsistent_anomaly_indices].sort_values(by='average_decision_score')

print('Inconsistent Anomalies with Sorted Decision Scores:')
print(sorted_inconsistent_anomalies[['average_decision_score']])


sorted_inconsistent_anomaly_indices = sorted_inconsistent_anomalies.index
sorted_inconsistent_anomaly_scores = decision_scores_df.loc[sorted_inconsistent_anomaly_indices]

print('Decision Scores of Inconsistent Anomalies for Each Run (Sorted):')
print(sorted_inconsistent_anomaly_scores)


    # Print the information for the specific row
specific_row_index = 1856

if specific_row_index in data.index:
    specific_row_count = anomaly_counts.loc[specific_row_index, 'count']
    specific_row_scores = decision_scores_df.loc[specific_row_index]


    print(f'\nInformation for row index {specific_row_index}:')
    print(f'Count of being marked as anomaly: {specific_row_count}')
    print(f'Decision scores across runs:\n{specific_row_scores}')
else:
    print(f'Row index {specific_row_index} not found in the dataset.')