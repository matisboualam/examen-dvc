import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

src_path = '/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/data/processed_data/'

# Separate features and target variable from test data
X_test = pd.read_csv(src_path+'X_test_scaled.csv')
y_test = pd.read_csv(src_path+'y_test.csv').values.ravel()

# Load the trained Random Forest model from a pickle file
with open('/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/models/trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred, name='silica_concentrate')
y_pred.to_csv(src_path+'predictions.csv', index=False)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save evaluation metrics to a JSON file
metrics = {'Mean Squared Error': mse, 'R2 Score': r2}
with open('/home/matis/Documents/cours/modules/dvc_dagshub/evaluation/examen-dvc/metrics/scores.json', 'w') as json_file:
    json.dump(metrics, json_file)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')