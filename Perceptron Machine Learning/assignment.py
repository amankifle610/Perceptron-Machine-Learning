import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        self.learning_rate = 0.01
        self.n_iterations = 1000

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                activation = np.dot(X[i], self.weights) + self.bias
                predicted = self.step_function(activation)
                
                self.weights += (self.learning_rate * (y[i] - predicted)) * X[i]
                self.bias += (self.learning_rate * (y[i] - predicted))

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        return np.array([self.step_function(np.dot(x, self.weights) + self.bias) for x in X])

df = pd.read_csv('census.csv')

for i, col in enumerate(df.columns):
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron()
perceptron.train(X_train, y_train)


y_pred = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")

conf_mat = confusion_matrix(y_test, y_pred)
labels = ['<=50K', '>50K']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()