import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.inspection import DecisionBoundaryDisplay

# Load Filtered Data
data_filtered = pd.read_csv('combined.csv')

# Separate Features and Apply t-SNE
features = data_filtered.columns[:-2]  # Exclude 'anomaly_score' and 'anomaly'
X = data_filtered[features].values

# Apply t-SNE to reduce to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

# Train the Isolation Forest on Reduced Data
clf = IsolationForest(contamination=0.1095759379317675, random_state=42)
clf.fit(X_reduced)

#Plot the Decision Boundary
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X_reduced,
    response_method="decision_function",
    alpha=0.5,
)

# Plot the scatter of the filtered data
labels = clf.predict(X_reduced)
disp.ax_.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, s=20, edgecolor="k", cmap='coolwarm')

# Add title
disp.ax_.set_title("Binary decision boundary of IsolationForest with t-SNE")
plt.axis("square")

# Manually create legend with colors from the colormap
import matplotlib.patches as mpatches
inlier_patch = mpatches.Patch(color='red', label='Inliers')
outlier_patch = mpatches.Patch(color='blue', label='Outliers')
plt.legend(handles=[inlier_patch, outlier_patch], title="Prediction")

# Show the plot
plt.show()
