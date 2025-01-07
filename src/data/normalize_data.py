import pandas as pd
from sklearn.preprocessing import MinMaxScaler

src_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data/'
dst_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data/'

# Load the data
X_train = pd.read_csv(src_path+'X_train.csv')
X_test = pd.read_csv(src_path+'X_test.csv')

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the normalized data back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save the normalized data
X_train_scaled.to_csv(dst_path+'X_train_scaled.csv', index=False)
X_test_scaled.to_csv(dst_path+'X_test_scaled.csv', index=False)