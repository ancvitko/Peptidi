import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
df = pd.read_csv('encoded_properties.csv')

# Check for missing values and fill them if necessary
df.fillna(df.mean(), inplace=True)

# Features for Isolation Forest (excluding any labels if present)
data = df.drop(columns=['sequence', 'label'], errors='ignore')

X_train, X_test = train_test_split(data, test_size=0.2, random_state=25)

# Ensure all attributes equally contribute
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# param_grid = {
#     'max_samples': ['auto'] + list(range(1, 1000, 100)),  # 'auto' or integer value between 1 and number of samples
#     'bootstrap': [True, False],  # True or False
#     'random_state': list(range(51)),  # Integer value between 0 and 100
#     'contamination': [round(x, 2) for x in list(np.arange(0.01, 0.51, 0.01))],  # Float value in range (0, 0.5]
#     'n_estimators': list(range(10, 1000, 100)) # Integer value between 10 and half of data.shape[0]
# }

# iso_forest1 = IsolationForest()
# # Perform grid search
# grid_search = GridSearchCV(iso_forest1, param_grid, cv=5, scoring='accuracy')

# grid_search.fit(X=data, y=y)

# # Best parameters
# best_params = grid_search.best_params_
# print("Best parameters: ", best_params)

# Train model with best parameters
# iso_forest = IsolationForest(**best_params)
# iso_forest.fit(features)


# Initialize the Isolation Forest model
#best so far
iso_forest = IsolationForest(bootstrap=True, max_samples=1024 ,contamination=0.25, random_state=25)
#iso_forest = IsolationForest(random_state=25, bootstrap=True, contamination=0.0001, max_features=10, max_samples=10, n_estimators=1000, n_jobs=-1)


# Fit the model on the data
iso_forest.fit(data)

# Predict anomalies
y_pred = iso_forest.predict(data)

decision_scores = iso_forest.decision_function(data)
df['score'] = decision_scores

# # -1 indicates an anomaly, 1 indicates a normal point => map so 0 is normal and 1 is anomaly
df['anomaly'] = y_pred
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

df.to_csv('results.csv', index=False)

# sort data from lowest score (most likely anomaly) to highest score
sorted_data = df.sort_values(by='score').reset_index(drop=True)

# Find the index where the anomaly score changes from 1 to 0
change_index = None
for i in range(1, len(sorted_data)):
    if sorted_data.loc[i-1, 'anomaly'] == 1 and sorted_data.loc[i, 'anomaly'] == 0:
        change_index = i
        break

if change_index is not None:
    print(f'Change Index: {change_index}')
    print('Row where the anomaly score changes from 1 to 0:')
    print(sorted_data.loc[change_index])
    
    print('Row before that:')
    print(sorted_data.loc[change_index - 1])
    
    print('Row after that:')
    print(sorted_data.loc[change_index + 1])
else:
    print('No change from anomaly (1) to normal (0) found.')

# # Save the anomalies to a new CSV file
# anomalies_df.to_csv('anomalies.csv', index=False)# Save the DataFrame with anomaly column to a CSV file
# df.to_csv('encoded_properties_with_anomalies.csv', index=False)

# # Filter the DataFrame to show only the anomalies
# anomalies_df = df[df['anomaly'] == -1]

# # Save the anomalies to a new CSV file
# anomalies_df.to_csv('anomalies.csv', index=False)

# # Read the CSV file
# df1 = pd.read_csv('peptides_data.csv')
# df2 = pd.read_csv('encoded_properties_with_anomalies.csv')

# # Count the occurrences of label "1" in the label column
# num_ones = df1['label'].eq(1).sum()
# sequence1 = df1['label']
# sequence2 = df2['anomaly']

# total_rows = df1.shape[0]

# correct_count_normal_instance = 0
# correct_count_anomalies = 0

# for value1, value2 in zip(sequence1, sequence2):
#     if (value1 == 0 and value2 == 1):
#         correct_count_normal_instance += 1
#     elif (value1 == 1 and value2 == -1):
#         correct_count_anomalies += 1
        
# print("Actual number of initial dataset anomalies",num_ones)
# print("Normal Instances: ", correct_count_normal_instance)
# print("Anomalies: ", correct_count_anomalies)
# print("Total number of correct anomalies and normal instances percentage: ", (correct_count_anomalies + correct_count_normal_instance)/total_rows)
