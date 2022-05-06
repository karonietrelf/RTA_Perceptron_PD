from Perceptron import Perceptron
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import pickle


# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)

y = df.iloc[0:100, 4].values

# Convert 0/1 values into -1/1 values (versicolor and setosa respectively)
y = np.where(y == 0, -1, 1)

# Extract `sepal length` and `petal length` from iris
X = df.iloc[0:100, [0, 2]].values

# Train model
model = Perceptron(0.1, 10)
model.fit(X, y)

if __name__ == "__main__":
    # Save model to the file
    with open('p-model.pkl', 'wb') as file:
        pickle.dump(model, file)
