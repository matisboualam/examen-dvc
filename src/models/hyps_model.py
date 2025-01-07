import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle

# Charger les données normalisées
X_train_scaled = pd.read_csv('/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data/y_train.csv')

# Initialiser le modèle
model = RandomForestRegressor()

# Définir la grille de paramètres à tester
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialiser GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Effectuer la recherche en grille
grid_search.fit(X_train_scaled, y_train.values.ravel())

# Extraire les meilleurs paramètres
best_params = grid_search.best_params_

# Sauvegarder les meilleurs paramètres dans un fichier .pkl
with open('/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/models/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print("Meilleurs paramètres trouvés :", best_params)