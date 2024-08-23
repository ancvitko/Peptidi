import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay

# Step 1: Load Filtered Data
data_filtered = pd.read_csv('filtered_train_peptides_label_0.csv')

# Select two features for visualization
# Example: Using 'Cruciani_1' and 'Cruciani_2' for the plot
X = data_filtered[['Cruciani_1', 'Cruciani_2']].values

# Train the Isolation Forest
clf = IsolationForest(contamination='auto', random_state=42)
clf.fit(X)

# Plot the Decision Boundary
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict",
    alpha=0.5,
)

# Plot the scatter of the filtered data
# Optionally, you can color points based on whether they are inliers or outliers according to the model
labels = clf.predict(X)
colors = ['red' if label == -1 else 'blue' for label in labels]
disp.ax_.scatter(X[:, 0], X[:, 1], c=colors, s=20, edgecolor="k")

# Add title and legend
disp.ax_.set_title("Binary decision boundary of IsolationForest | Cruciani 1 and Cruciani 2")
plt.axis("square")

# Manually create legend
import matplotlib.patches as mpatches
inlier_patch = mpatches.Patch(color='blue', label='Inliers')
outlier_patch = mpatches.Patch(color='red', label='Outliers')
plt.legend(handles=[inlier_patch, outlier_patch], title="Prediction")

# Show the plot
plt.show()
