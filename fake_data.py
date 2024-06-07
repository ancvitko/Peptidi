import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

data = pd.read_csv('encoded_properties.csv')
X = data.drop(columns=['label'], errors='ignore')
print(X)

ranges = {
    'Cruciani_1': (-5, -2), # usually -0.91 ~ 1
    'Cruciani_2': (-8, -2), # usually -0.91 ~ 0.81
    'Cruciani_3': (1, 10), # usually -1 ~ 0.86
    'Instability_Index': (483, 600), # usually -43.43 ~ 483
    'Boman_Index': (-30, -5), # usually -3.823 ~ 12.76
    'Hydrophobicity_Eisenberg': (-20, -4), # usually -2.1 ~ 1
    'Hydrophobic_Moment': (2, 5), # usually 0.008 ~ 1.185
    'Aliphatic_Index': (300, 350), # usually 0 ~ 292.5
    'Isoelectric_Point_Lehninger': (-5, 1), # usually 2.185 ~ 13.802
    'Charge_pH7.4_Lehninger': (30, 100), # usually -20 ~ 28
    'Freq_Tiny': (0, 1),
    'Freq_Small': (0, 1),
    'Freq_Aliphatic': (0, 1),
    'Freq_Aromatic': (0, 1),
    'Freq_Non_polar': (0, 1),
    'Freq_Polar': (0, 1),
    'Freq_Charged': (0, 1),
    'Freq_Basic': (0, 1),
    'Freq_Acidic': (0, 1),
}

def generate_fake_data(ranges, num_samples):
    fake_data = pd.DataFrame()
    for column, (min_val, max_val) in ranges.items():
        fake_data[column] = np.random.uniform(low=min_val, high=max_val, size=num_samples)
    return fake_data

# Generate fake data
num_fake_samples = 2000
fake_df = generate_fake_data(ranges, num_fake_samples)

# Add a label to indicate fake data
data['is_fake'] = 0
fake_df['is_fake'] = 1

print(fake_df)
# Combine original and fake data
combined_data = pd.concat([data, fake_df], ignore_index=True)
print("combined data:")
print(combined_data)
# Separate features and labels
X_combined = combined_data.drop(columns=['label', 'is_fake'], errors='ignore')
y_combined = combined_data['is_fake']

# Isolation Forest
clf = IsolationForest(contamination=0.18, random_state=42)
clf.fit(X_combined)

# Get anomaly labels and decision scores
anomaly_labels = clf.predict(X_combined)
decision_scores = clf.decision_function(X_combined)

combined_data['anomaly_label'] = anomaly_labels
combined_data['decision_score'] = decision_scores

# Check if fake data was detected as anomalies
fake_data_anomalies = combined_data[combined_data['is_fake'] == 1]

print('Fake Data Anomalies:')
print(fake_data_anomalies[['anomaly_label', 'decision_score', 'is_fake']])

num_fake_anomalies_detected = fake_data_anomalies[fake_data_anomalies['anomaly_label'] == -1].shape[0]
print(f'Number of fake data points detected as anomalies: {num_fake_anomalies_detected} out of {num_fake_samples}')
