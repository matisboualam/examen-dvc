import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

src_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data/'

# Load the best parameters from a pickle file
with open('/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

# Load the training and testing data from CSV files
X_train = pd.read_csv(src_path+'X_train_scaled.csv')
y_train = pd.read_csv(src_path+'y_train.csv').values.ravel()

# Initialize the model with the best parameters
model = RandomForestRegressor(**best_params)

# Train the model
model.fit(X_train, y_train)

# Export the trained model weights to a pickle file
with open('/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
