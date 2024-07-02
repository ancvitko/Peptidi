import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
data = pd.read_csv("filtered_properties.csv")

# Display the first few rows of the data to understand its structure
print(data.head())


# Choose the attribute to plot
attribute_to_plot = 'anomaly_score'

# Find the minimum and maximum values of the chosen attribute
min_value = data[attribute_to_plot].min()
max_value = data[attribute_to_plot].max()

# Generate Y-axis values (all the same) for plotting
y_value = 0  # Fixed Y-axis value
y_values = [y_value] * len(data[attribute_to_plot])

# Plot the chosen attribute as dots along the X-axis range
plt.figure(figsize=(10, 6))
plt.scatter(data[attribute_to_plot], y_values, color='b', label=attribute_to_plot)

# Add lines indicating min and max values
plt.axvline(x=min_value, color='r', linestyle='--', label='Min Value')
plt.axvline(x=max_value, color='g', linestyle='--', label='Max Value')

# Add labels and title
plt.xlabel(attribute_to_plot)
plt.ylabel('Fixed Y-axis Value')
plt.title(f'{attribute_to_plot} Values Over X-Axis Range')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
