# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.model_selection import train_test_split

# def process_file(input_file, train_output_file, test_output_file):
#     # Load Data
#     data = pd.read_csv(input_file)

#     # Split Data into Training and Test Sets
#     train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#     # Train Isolation Forest on Training Data
#     iso_forest = IsolationForest(contamination='auto', random_state=42)
#     iso_forest.fit(train_data)

#     # Add anomaly scores to the Training DataFrame
#     train_data['anomaly_score'] = iso_forest.decision_function(train_data)

#     # Identify Outliers in the Training Data
#     train_data['anomaly'] = iso_forest.predict(train_data.drop(columns=['anomaly_score']))

#     # Sort training data by anomaly score in ascending order (worst scores first)
#     train_data_sorted = train_data.sort_values(by='anomaly_score', ascending=True)

#     # Remove Outliers from Training Data
#     outlier_fraction = 0.1  # Define the proportion of outliers to remove
#     num_outliers = int(len(train_data_sorted) * outlier_fraction)
#     train_data_filtered = train_data_sorted.iloc[num_outliers:]

#     # Save Filtered Training Data and Test Data
#     train_data_filtered.to_csv(train_output_file, index=False)
#     test_data.to_csv(test_output_file, index=False)

#     print(f"Processed {input_file}.")
#     print(f"Filtered training data saved to {train_output_file}.")
#     print(f"Test data saved to {test_output_file}.")

# if __name__ == "__main__":
#     process_file('encoded_peptides_label_0.csv', 'filtered_train_peptides_label_0.csv', 'test_peptides_label_0.csv')
#     process_file('encoded_peptides_label_1.csv', 'filtered_train_peptides_label_1.csv', 'test_peptides_label_1.csv')

#     print("Anomaly detection completed and files saved successfully.")


import pandas as pd
from sklearn.ensemble import IsolationForest

def process_file(input_file, output_file):
    # Load Data
    data = pd.read_csv(input_file)

    # Train Isolation Forest on Entire Data
    iso_forest = IsolationForest(contamination=0.1095759379317675, random_state=42)
    iso_forest.fit(data)

    # Add anomaly scores to the DataFrame
    data['anomaly_score'] = iso_forest.decision_function(data)

    # Identify Outliers in the Data
    data['anomaly'] = iso_forest.predict(data.drop(columns=['anomaly_score']))

    # Sort data by anomaly score in ascending order (worst scores first)
    data_sorted = data.sort_values(by='anomaly_score', ascending=True)

    # # Remove Outliers from Data
    # if input_file == 'encoded_peptides_label_0.csv':
    #     outlier_fraction = 0.1041  # Define the proportion of outliers to remove
    # else:
    #     outlier_fraction = 0.1182  # Define the proportion of outliers to remove

    outlier_fraction = 0
    num_outliers = int(len(data_sorted) * outlier_fraction)
    data_filtered = data_sorted.iloc[num_outliers:]

    # Save the Filtered Data
    data_filtered.to_csv(output_file, index=False)

    print(f"Processed {input_file}.")
    print(f"Filtered data saved to {output_file}.")

if __name__ == "__main__":
    process_file('encoded_peptides_label_0.csv', 'filtered_peptides_label_0.csv')
    process_file('encoded_peptides_label_1.csv', 'filtered_peptides_label_1.csv')

    print("Anomaly detection completed and files saved successfully.")


    # Load the filtered datasets
    data_label_0 = pd.read_csv('filtered_peptides_label_0.csv')
    data_label_1 = pd.read_csv('filtered_peptides_label_1.csv')

    # Count the -1s and 1s in the 'anomaly' column for label 0
    value_counts_label_0 = data_label_0['anomaly'].value_counts()
    count_minus1_label_0 = value_counts_label_0.get(-1, 0)  # Default to 0 if -1 is not present
    count_1_label_0 = value_counts_label_0.get(1, 0)  # Default to 0 if 1 is not present

    # Count the -1s and 1s in the 'anomaly' column for label 1
    value_counts_label_1 = data_label_1['anomaly'].value_counts()
    count_minus1_label_1 = value_counts_label_1.get(-1, 0)  # Default to 0 if -1 is not present
    count_1_label_1 = value_counts_label_1.get(1, 0)  # Default to 0 if 1 is not present

    # Print the counts
    print(f"Label 0: -1s={count_minus1_label_0}, 1s={count_1_label_0}")
    print(f"Label 1: -1s={count_minus1_label_1}, 1s={count_1_label_1}")
