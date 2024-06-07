from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import peptides
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("peptides_data.csv")

sequences = data["sequence"].to_numpy()
y = data["label"].to_numpy()

groups = (
    ('A', 'C', 'G', 'S', 'T'),                                  # Tiny
    ('A', 'C', 'D', 'G', 'N', 'P', 'S', 'T', 'V'),              # Small 
    ('A', 'I', 'L', 'V'),                                       # Aliphatic
    ('F', 'H', 'W', 'Y'),                                       # Aromatic
    ('A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W', 'Y'),    # Non-polar
    ('D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T'),              # Polar
    ('D', 'E', 'H', 'K', 'R'),                                  # Charged
    ('H', 'K', 'R'),                                            # Basic
    ('D', 'E')                                                  # Acidic
)

X = []
for sequence in sequences:
    sequence = sequence.upper()

    peptide = peptides.Peptide(sequence)
    x = [
        peptide.cruciani_properties()[0],
        peptide.cruciani_properties()[1],
        peptide.cruciani_properties()[2],
        peptide.instability_index(),
        peptide.boman(),
        peptide.hydrophobicity("Eisenberg"),
        peptide.hydrophobic_moment(angle=100, window=min(len(sequence), 11)),
        peptide.aliphatic_index(),
        peptide.isoelectric_point("Lehninger"),
        peptide.charge(pH=7.4, pKscale="Lehninger"),
    ]

    # Count tiny, small, aliphatic, ..., basic and acidic amino acids
    for group in groups:
         count = 0
         for amino in group:
             count += sequence.count(amino)
         x.append(count)
         x.append(count / len(sequence))
    X.append(x)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {'n_estimators': [1000, 1500], 
              'max_samples': [10], 
              'contamination': ['auto', 0.0001, 0.0002], 
              'max_features': [10, 15], 
              'bootstrap': [True], 
              'n_jobs': [-1]}

model = IsolationForest(random_state=25, bootstrap=True, contamination=0.0001, max_features=10, max_samples=10, n_estimators=1000, n_jobs=-1)

model.fit(X_train, y_train)
predict = model.predict(X_test, y_test)

# grid_search = model_selection.GridSearchCV(model, 
#                                            param_grid,
#                                            scoring="neg_mean_squared_error", 
#                                            refit=True,
#                                            cv=10, 
#                                            return_train_score=True)
# grid_search.fit(X_train, y_train)

# best_model = grid_search.fit(X_train, y_train)
# print('Optimum parameters', best_model.best_params_)

models = {
    # 'Linear Regression': LinearRegression(),
    # 'Ridge Regression': Ridge(),
    # 'Lasso Regression': Lasso(),
    # 'Support Vector Regression': SVR(),
    # 'K-Nearest Neighbors': KNeighborsRegressor(),
    # 'Logistic Regression': LogisticRegression(),
    # 'Random Forest': RandomForestClassifier(),
    # 'Support Vector Machine': SVC(probability=True),
    # 'Gradient Boosting': GradientBoostingClassifier(),
    'Isolation Forest': IsolationForest()
}

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     #y_prob = model.predict_proba(X_test)[:, 1]
    
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     #roc_auc = roc_auc_score(y_test, y_prob)
    
#     print(f'{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')