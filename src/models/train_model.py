import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os

src_path_1 = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/preprocessed_data/'
src_path_2 = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data/'
dst_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/models'
os.makedirs(dst_path, exist_ok=True)

with open('/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/hyps/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

X_train = pd.read_csv(src_path_2+'X_train_scaled.csv')
y_train = pd.read_csv(src_path_1+'y_train.csv').values.ravel()

model = RandomForestRegressor(**best_params)

model.fit(X_train, y_train)

with open(dst_path+'/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
