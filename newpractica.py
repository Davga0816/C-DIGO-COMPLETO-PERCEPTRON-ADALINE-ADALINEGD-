import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            errors = y - self.net_input(X)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, 0)

# Cargar el conjunto de datos
df = pd.read_csv('creditcard.csv')

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Escalar características para el entrenamiento
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Entrenar el modelo Adaline
ada = AdalineGD(eta=0.01, n_iter=50)
ada.fit(X_train_std, y_train)

# Realizar predicciones
y_pred = ada.predict(X_test_std)

# Calcular la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)

# Personalizar la gráfica de convergencia del costo
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o', color='b', linestyle='-')
plt.title('Convergencia del Costo del Modelo Adaline')
plt.xlabel('Épocas')
plt.ylabel('Costo')
plt.grid(True)  # Agregar una cuadrícula

# Agregar una leyenda
plt.legend(['Entrenamiento'], loc='upper right')

plt.show()
