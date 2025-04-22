import numpy as np
from matplotlib import pyplot as plt

class Perceptron:
    def __init__(self, n_input, n_hidden, n_output):
        self.w1 = np.random.randn(n_input, n_hidden)
        self.b1 = np.zeros((1, n_hidden))
        self.w2 = np.random.randn(n_hidden, n_output)
        self.b2 = np.zeros((1, n_output))

    def forward(self, X):
        self.Z1 = np.dot(X, self.w1) + self.b1
        self.A1 = np.tanh(self.Z1)#скрытый слой гиперболического тангенса
        self.Z2 = np.dot(self.A1, self.w2) + self.b2
        self.A2 = self.sigmoid(self.Z2) #выходной слой (A2) с помощью сигмоидной функции.
        return self.A2

    def sigmoid(self, Z): #сигмоиду от Z.
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z): #производная
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.w2.T) * (1 - np.power(self.A1, 2))
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        self.w2 = self.w2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2
        self.w1 = self.w1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fxh1 = f(x)
        x[ix] = old_value - h
        fxh2 = f(x)
        grad[ix] = (fxh1 - fxh2) / (2 * h)
        x[ix] = old_value
        it.iternext()
    return grad


def loss_function(model, X, y):
    # функция ошибки
    A2 = model.forward(X)
    return np.mean(-y * np.log(A2) - (1 - y) * np.log(1 - A2))


def test():
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 1, 0, 0])
    n_input = X.shape[1]
    n_hidden = 2
    n_output = 1
    model = Perceptron(n_input, n_hidden, n_output)
    # вычисляем градиент при помощи обратного распространения ошибки
    A2 = model.forward(X)
    model.backward(X, y.reshape(-1, 1), 0.1)
    dW1, db1, dW2, db2 = model.w1, model.b1, model.w2, model.b2
    # вычисляем градиент численно
    params = [model.w1, model.b1, model.w2, model.b2]
    grads = [numerical_gradient(lambda p: loss_function(model, X, y), p) for p in params]
    dW1_num, db1_num, dW2_num, db2_num = grads
    # сравниваем градиенты
    print(f"dW1 error: {np.mean(np.abs(dW1 - dW1_num))}")
    print(f"db1 error: {np.mean(np.abs(db1 - db1_num))}")
    print(f"dW2 error: {np.mean(np.abs(dW2 - dW2_num))}")
    print(f"db2 error: {np.mean(np.abs(db2 - db2_num))}")


def plot_all(model):
    X_test = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_pred = np.round(model.forward(X_test)).astype(int)
    colors = ['red' if y == 0 else 'blue' for y in y_pred]
    y_origin = np.array([[1], [1], [0], [0]])
    colors1 = ['red' if y == 0 else 'blue' for y in y_origin]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], color=colors1)
    plt.title('Origin')
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], color=colors)
    plt.title('Predictions')
    plt.show()


def main():
    X = np.array([[1, 0],
                  [0, 1],
                  [1, 1],
                  [0, 0]])
    y = np.array([[1], [1], [0], [0]])
    n_input = 2
    n_hidden = 3
    n_output = 1
    model = Perceptron(n_input, n_hidden, n_output)
    epochs = 5000
    learning_rate = 0.1
    losses = []
    for i in range(epochs):
        A2 = model.forward(X)
        model.backward(X, y, learning_rate)
        loss = np.mean(-y * np.log(A2) - (1 - y) * np.log(1 - A2))
        losses.append(loss)
        if i % 1000 == 0:
            print(f"Loss after iteration {i}: {loss}")
    print(f"Predictions: {A2}")
    plt.plot(losses)
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.title('Значение функции потерь в течение обучения')
    plt.show()
    plot_all(model)
    test()


if __name__ == '__main__':
    main()

#таблица xor и что это
#
