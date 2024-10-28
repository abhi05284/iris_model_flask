from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
dump(model, 'iris_model.joblib')
