import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Load the encoded data
data = pd.read_csv('encoded_properties.csv')

# Assuming the label column is named 'label' in the original dataset and is added to the encoded dataframe
original_data = pd.read_csv('peptides_data.csv')
labels = original_data['label']

# Ensure the encoded data and labels have the same length
assert len(data) == len(labels), "Mismatch between encoded data and labels length"

# Apply Isolation Forest
isolation_forest = IsolationForest(
    
    max_features=1,
    contamination=0.394,
    max_samples=2048,
    random_state=42)
# Fit the model
isolation_forest.fit(data)

# Predict anomalies
predictions = isolation_forest.predict(data)

# Convert predictions: 1 (normal) to 0, -1 (anomaly) to 1
anomaly_labels = [0 if x == 1 else 1 for x in predictions]

# Add the anomaly predictions to the dataframe
data['anomaly'] = anomaly_labels
data['label'] = labels

# Evaluate the model
print("Classification Report:")
print(classification_report(labels, anomaly_labels))
print("Confusion Matrix:")
print(confusion_matrix(labels, anomaly_labels))

# Save the results to a new CSV file
data.to_csv('encoded_properties_with_anomalies.csv', index=False)
