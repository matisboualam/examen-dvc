import pandas as pd
import os
from sklearn.model_selection import train_test_split

src_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/raw_data/'
dst_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/preprocessed_data'
os.makedirs(dst_path, exist_ok=True)

data = pd.read_csv(src_path+'raw.csv')

target_column = 'silica_concentrate'

Y = data[target_column]
X = data.drop(columns=[target_column])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train['date'] = pd.to_datetime(X_train['date'])
X_test['date'] = pd.to_datetime(X_test['date'])

X_train['year'] = X_train['date'].dt.year
X_train['month'] = X_train['date'].dt.month
X_train['day'] = X_train['date'].dt.day
X_train['dayofweek'] = X_train['date'].dt.dayofweek

X_test['year'] = X_test['date'].dt.year
X_test['month'] = X_test['date'].dt.month
X_test['day'] = X_test['date'].dt.day
X_test['dayofweek'] = X_test['date'].dt.dayofweek

X_train = X_train.drop(columns=['date'])
X_test = X_test.drop(columns=['date'])

X_train.to_csv(dst_path+'/X_train.csv', index=False)
X_test.to_csv(dst_path+'/X_test.csv', index=False)
y_train.to_csv(dst_path+'/y_train.csv', index=False)
y_test.to_csv(dst_path+'/y_test.csv', index=False)