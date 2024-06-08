import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Used to generate the plots in 'images' folder and get statistics of input dataset
df = pd.read_csv('peptides_data.csv')
y = df['label'].values

column_values = df.iloc[:, 0]

# Calculate the length of each string
lengths = column_values.str.len()

# Compute the average length
average_length = lengths.mean()

print("Average length of strings in the first column:", average_length)

print("Number of anomalies: ", df['label'].eq(1).sum())
print("Percentage of anomalies: ", (df['label'].eq(1).sum() / len(df)) * 100)

X = pd.read_csv('encoded_properties.csv')

# for column in X.columns:
#     # Create a new figure for each plot
#     plt.figure(figsize=(8, 6))
    
#     # Plot the data for the current column
#     plt.plot(X[column], marker='o', linestyle='-')
    
#     # Set plot title and labels
#     plt.title(f'Property: {column}')
#     plt.xlabel('Peptides')
#     plt.ylabel('Value')
    
#     # Optionally, save the plot to a file
#     plt.savefig(f'{column}_plot.png')
    

stats = X.describe(percentiles=[.25, .75]).transpose()

# Calculate the IQR (Interquartile Range)
stats['IQR'] = stats['75%'] - stats['25%']


# Calculate the lower whisker
stats['Lower Whisker'] = stats['25%'] - 1.5 * stats['IQR']
# Ensure lower whisker does not go below the minimum value
stats['Lower Whisker'] = stats[['Lower Whisker', 'min']].max(axis=1)

# Calculate the upper whisker
stats['Upper Whisker'] = stats['75%'] + 1.5 * stats['IQR']
# Ensure upper whisker does not go above the maximum value
stats['Upper Whisker'] = stats[['Upper Whisker', 'max']].min(axis=1)

# Select relevant columns
stats = stats[['min', 'max', 'mean', '50%', '25%', '75%', 'Lower Whisker', 'Upper Whisker']]
stats.columns = ['Min', 'Max', 'Mean', 'Median', 'Lower Quartile', 'Upper Quartile', 'Lower Whisker', 'Upper Whisker']

print(stats)

plt.show()
