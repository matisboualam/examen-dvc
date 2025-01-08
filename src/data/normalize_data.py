import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

src_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/preprocessed_data/'
dst_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data'
os.makedirs(dst_path, exist_ok=True)

X_train = pd.read_csv(src_path+'X_train.csv')
X_test = pd.read_csv(src_path+'X_test.csv')

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train_scaled.to_csv(dst_path+'/X_train_scaled.csv', index=False)
X_test_scaled.to_csv(dst_path+'/X_test_scaled.csv', index=False)
