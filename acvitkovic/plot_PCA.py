import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

# Step 1: Load Filtered Data
data_filtered = pd.read_csv('filtered_peptides.csv')

# Step 2: Separate Features and Apply PCA
features = data_filtered.columns[:-2]  # Exclude 'anomaly_score' and 'anomaly'
X = data_filtered[features].values

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Step 3: Train the Isolation Forest on Reduced Data
clf = IsolationForest(contamination='auto', random_state=42)
clf.fit(X_reduced)

# Step 4: Plot the Decision Boundary
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X_reduced,
    response_method="predict",
    alpha=0.5,
)

# Plot the scatter of the filtered data
labels = clf.predict(X_reduced)
disp.ax_.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, s=20, edgecolor="k", cmap='coolwarm')

# Add title and legend
disp.ax_.set_title("Binary decision boundary of IsolationForest with PCA")
plt.axis("square")
handles, _ = disp.ax_.get_legend_handles_labels()
disp.ax_.legend(handles=handles, labels=["inliers", "outliers"], title="Prediction")

# Manually create legend with colors from the colormap
import matplotlib.patches as mpatches
inlier_patch = mpatches.Patch(color='red', label='Inliers')
outlier_patch = mpatches.Patch(color='blue', label='Outliers')
plt.legend(handles=[inlier_patch, outlier_patch], title="Prediction")

# Show the plot
plt.show()
