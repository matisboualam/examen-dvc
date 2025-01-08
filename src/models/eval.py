import pandas as pd
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

src_path_1 = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/preprocessed_data/'
src_path_2 = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data/'
dst_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/metrics'
os.makedirs(dst_path, exist_ok=True)

X_test = pd.read_csv(src_path_2+'X_test_scaled.csv')
y_test = pd.read_csv(src_path_1+'y_test.csv').values.ravel()

with open('/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/models/trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred, name='silica_concentrate')
y_pred.to_csv(dst_path+'/predictions.csv', index=False)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {'Mean Squared Error': mse, 'R2 Score': r2}
with open(dst_path+'/scores.json', 'w') as json_file:
    json.dump(metrics, json_file)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
