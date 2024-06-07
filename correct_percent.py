import pandas as pd

# Read the CSV file
df1 = pd.read_csv('peptides_data.csv')
df2 = pd.read_csv('encoded_properties_with_anomalies.csv')

# Count the occurrences of label "1" in the label column
num_ones = df1['label'].eq(1).sum()
sequence1 = df1['label']
sequence2 = df2['anomaly']

total_rows = df1.shape[0]

correct_count_normal_instance = 0
correct_count_anomalies = 0

for value1, value2 in zip(sequence1, sequence2):
    if (value1 == 0 and value2 == 1):
        correct_count_normal_instance += 1
    elif (value1 == 1 and value2 == -1):
        correct_count_anomalies += 1
print("Actual number of initial dataset anomalies",num_ones)
print("Normal Instances: ", correct_count_normal_instance)
print("Anomalies: ", correct_count_anomalies)
print("Total number of correct anomalies and normal instances percentage: ", (correct_count_anomalies + correct_count_normal_instance)/total_rows)