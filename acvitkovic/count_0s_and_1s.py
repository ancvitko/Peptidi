# Import pandas
import pandas as pd

# Load the filtered datasets
data_label_0 = pd.read_csv('filtered_peptides_label_0.csv')
data_label_1 = pd.read_csv('filtered_peptides_label_1.csv')

# Count the 0s and 1s in the 'anomaly' column
count_minus1_label_0 = data_label_0['anomaly'].value_counts()[-1]
count_1_label_0 = data_label_0['anomaly'].value_counts()[1]
count_minus1_label_1 = data_label_1['anomaly'].value_counts()[-1]
count_1_label_1 = data_label_1['anomaly'].value_counts()[1]

# Print the counts
print(f"Label 0: -1s={count_minus1_label_0}, 1s={count_1_label_0}")
print(f"Label 1: -1s={count_minus1_label_1}, 1s={count_1_label_1}")