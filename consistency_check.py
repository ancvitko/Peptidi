import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

data = pd.read_csv('encoded_properties.csv')
X = data.drop(columns=['sequence', 'label'], errors='ignore')

iterations_results = []
specific_row_index = 4739
# Define the specific index you're interested in

# Function to run Isolation Forest and get anomaly indices
def get_anomalies_with_scores(X, random_state, cont, max_samples):
    clf = IsolationForest(contamination=cont, random_state=random_state, max_samples=max_samples)
    clf.fit(X)
    #-1 for anomaly, 1 for normal
    anomalies = clf.predict(X)
    anomaly_indices = np.where(anomalies == -1)[0]
    decision_scores = clf.decision_function(X)
    
    return anomaly_indices, decision_scores, anomalies

num_runs = 10
all_anomalies = []
all_scores = []
all_labels = []
change_indices = []
contaminations = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
samples = [256, 540, 745, 1024, 2048, 3328, 3800, 4352, 5120, 7459]

specific_row_data = []

for i in range(num_runs):
    anomalies, scores, labels = get_anomalies_with_scores(X, random_state=25, cont=0.10, max_samples=samples[i])
    all_anomalies.append(anomalies)
    all_scores.append(scores)
    all_labels.append(labels)
    
    # Collect data for the specific row before sorting
    if specific_row_index in data.index:
        specific_row_score = scores[specific_row_index]
        specific_row_label = labels[specific_row_index]
        specific_row_data.append((i, specific_row_score, specific_row_label))
    
    # Create a DataFrame for the current run
    run_data = data.copy()
    run_data['anomaly'] = labels
    run_data['score'] = scores
    
    # Number of anomalies
    num_anomalies = len(anomalies)
    
    # Percentage of anomalies in the whole dataset
    perc_anomalies = (len(anomalies) / len(X)) * 100
    
    # Anomaly with highest decision score (least abnormal)
    if num_anomalies > 0:
        highest_decision_score = np.max(scores[anomalies])
        lowest_decision_score = np.min(scores[anomalies])
    else:
        highest_decision_score = None
        lowest_decision_score = None
    
    # Store the iteration results
    iteration_info = {
        'iteration': i,
        'number_of_anomalies': num_anomalies,
        'percentage_of_anomalies': perc_anomalies,
        'highest_decision_score': highest_decision_score,
        'lowest_decision_score': lowest_decision_score
    }
    
    iterations_results.append(iteration_info)
    
    # Sort the DataFrame by the score column in ascending order
    sorted_run_data = run_data.sort_values(by='score').reset_index(drop=True)

# Get general information on each iteration
iterations_df = pd.DataFrame(iterations_results)
print("GENERAL INFO OF ITERATIONS: ")
print(iterations_df)

anomaly_counts = pd.DataFrame(0, index=X.index, columns=['count'])
decision_scores_df = pd.DataFrame(0, index=X.index, columns=[f'run_{i}' for i in range(num_runs)])

# Count the number of times each index is detected as an anomaly and store the decision scores
for run_index, (anomalies, scores) in enumerate(zip(all_anomalies, all_scores)):
    anomaly_counts.loc[anomalies, 'count'] += 1
    decision_scores_df[f'run_{run_index}'] = scores

# Add the average decision score to the main DataFrame
data['average_decision_score'] = decision_scores_df.mean(axis=1)

print(anomaly_counts)

# Filter rows that are consistently marked as anomalies
consistent_anomalies = anomaly_counts[anomaly_counts['count'] == num_runs]

consistent_anomaly_indices = consistent_anomalies.index
sorted_consistent_anomalies = data.loc[consistent_anomaly_indices].sort_values(by='average_decision_score')

print('Consistent Anomalies with Sorted Decision Scores:')
print(sorted_consistent_anomalies[['average_decision_score']])

sorted_consistent_anomaly_indices = sorted_consistent_anomalies.index
sorted_consistent_anomaly_scores = decision_scores_df.loc[sorted_consistent_anomaly_indices]

print('Decision Scores of Consistent Anomalies for Each Run (Sorted):')
print(sorted_consistent_anomaly_scores)

# Filter rows that are inconsistently marked as anomalies
inconsistent_anomalies = anomaly_counts[np.logical_and(anomaly_counts['count'] != num_runs, anomaly_counts['count'] > 0)]

inconsistent_anomaly_indices = inconsistent_anomalies.index

sorted_inconsistent_anomalies = data.loc[inconsistent_anomaly_indices].sort_values(by='average_decision_score')

# Sort from lowest to highest decision score
print('Inconsistent Anomalies with Sorted Decision Scores:')
print(sorted_inconsistent_anomalies[['average_decision_score']])

sorted_inconsistent_anomaly_indices = sorted_inconsistent_anomalies.index
sorted_inconsistent_anomaly_scores = decision_scores_df.loc[sorted_inconsistent_anomaly_indices]

print('Decision Scores of Inconsistent Anomalies for Each Run (Sorted):')
print(sorted_inconsistent_anomaly_scores)

# Print the information for the specific row before sorting
if specific_row_index in data.index:
    print(f'\nInformation for row index {specific_row_index} across all runs:')
    for run_id, score, label in specific_row_data:
        anomaly_status = "Anomaly" if label == -1 else "Normal"
        print(f'Run {run_id}: Score = {score:.6f}, Label = {anomaly_status}')
else:
    print(f'Row index {specific_row_index} not found in the dataset.')

indices_to_check = np.array([7218, 2205, 4739, 3818])

# Check if each index is in the inconsistent_anomalies DataFrame
for index in indices_to_check:
    if index in inconsistent_anomalies.index:
        print(f'Index {index} is present in the inconsistent anomalies.')
    else:
        print(f'Index {index} is NOT present in the inconsistent anomalies.')