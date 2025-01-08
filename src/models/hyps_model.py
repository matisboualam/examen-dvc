import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import os

src_path_1 = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/preprocessed_data/'
src_path_2 = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data/'
dst_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/hyps'
os.makedirs(dst_path, exist_ok=True)

X_train = pd.read_csv(src_path_2+'X_train_scaled.csv')
y_train = pd.read_csv(src_path_1+'y_train.csv')

model = RandomForestRegressor()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train.values.ravel())

best_params = grid_search.best_params_

with open(dst_path+'/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print("Meilleurs paramètres trouvés :", best_params)
