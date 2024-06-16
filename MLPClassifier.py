import numpy as np
import Activation as Act


def _calculate_accuracy(targets, predictions):
    return np.mean(np.array(targets) == np.array(predictions))


def _calculate_mse(targets, predictions):
    return np.mean((np.array(targets) - np.array(predictions)) ** 2)


def back_propagate(mlp, outputs, error, activation_func='sigmoid_der'):
    gradients = []
    delta = error
    for i in range(len(outputs) - 1, 0, -1):
        d_f = Act.activate(outputs[i], activation_func)
        gradient = np.dot(delta.T, outputs[i - 1])
        gradients.insert(0, gradient)

        if i > 1:
            delta = np.dot(delta, mlp.weights[i - 1]) * d_f

    return gradients


def _update_weights(mlp, gradients):
    for i in range(len(gradients)):
        mlp.weights[i] -= mlp.learning_rate_init * gradients[i]


List_AccTrain = []
List_MseTrain = []
List_AccValid = []
List_MseValid = []


def train(mlp, x_train, y_train, x_val, y_val):
    mlp.generate_weights()
    for epoch in range(mlp.epochs):
        for j in range(len(x_train)):
            inputs = x_train[j]  # shape input = (n0,)
            inputs = np.reshape(inputs, newshape=(1, mlp.layer_dimensions[0]))  # shape input = (1,n0)
            target = y_train[j]
            _, _, y3 = mlp.feedforward(inputs)
            error = target - y3
            gradients = back_propagate(mlp, inputs, error)
            _update_weights(mlp, gradients)
            mlp.epoch_counter += 1

        momentum = mlp.momentum
        if mlp.learn_rate == 'constant':
            pass
        elif mlp.learn_rate == 'adaptive':
            mlp.update_learning_rate(momentum)
            # print('*** Update: Learn rate decreased by 20% ***')

        tolerance = mlp.tolerance
        mse_train, acc_train = mlp.evaluate(x_train, x_train, tol=tolerance)
        mse_valid, acc_valid = mlp.evaluate(x_val, y_val, tol=tolerance)
        List_MseTrain.append(mse_train)
        List_AccTrain.append(acc_train)
        List_MseValid.append(mse_valid)
        List_AccValid.append(acc_valid)

        tol = mlp.tolerance
        if List_AccValid[epoch] - List_AccValid[epoch-1] < tol:
            print('Stopped because of tolerance')
            return

        def print_epochs():
            if mlp.verbose:
                for i in range(mlp.epoch_counter):
                    print(f'Epoch {i}:\n'
                          f' Train MSE = {mse_train}   Train Acc = {acc_train},\n'
                          f' Valid MSE = {mse_valid}   Valid Acc = {acc_valid},\n')
            else:
                print('Verbose if Off.')
                print(f'Epoch {mlp.epoch_counter}:\n'
                      f' Train MSE = {mse_train}   Train Acc = {acc_train},\n'
                      f' Valid MSE = {mse_valid}   Valid Acc = {acc_valid},\n')
        print_epochs()

        return List_AccTrain, List_MseTrain, List_AccValid, List_MseValid


class MLP:
    def __init__(self, layer_dimensions, activation='linear', epochs=500, learn_rate='constant',
                 learning_rate_init=0.01,
                 verbose=False,
                 tolerance=0.00001,
                 momentum=0.7):

        self.weights = []
        self.layer_dimensions = layer_dimensions
        self.num_layers = len(layer_dimensions)
        self.activation = activation
        self.back_propagate_function = None
        self.epochs = int(epochs)
        self.learning_rate_init = float(learning_rate_init)
        self.learn_rate = learn_rate
        self.verbose = verbose
        self.epoch_counter = 0
        self.tolerance = float(tolerance)
        self.momentum = momentum

    def generate_weights(self):
        for i in range(1, self.num_layers):
            self.weights.append(np.random.uniform(low=-10, high=+10,
                                                  size=(self.layer_dimensions[i], self.layer_dimensions[i-1])))

    # feedforward algorithm
    def feedforward(self, input_net):
        activation_func = self.activation
        outputs = [input_net]
        for i in range(self.num_layers - 1):
            x = np.dot(outputs[-1], self.weights[i].T)
            y = Act.activate(x, activation_func)
            outputs.append(y)
        return outputs[-3], outputs[-2], outputs[-1]

    def evaluate(self, inputs, targets, tol):
        net_outputs = []
        target_list = []
        rounded_net_outputs = []

        for i in range(len(inputs)):
            input_data = inputs[i]
            target_data = targets[i]
            _, _, pred = self.feedforward(input_data)
            net_outputs.append(pred)
            target_list.append(target_data)
            rounded_net_outputs.append(np.round(pred))

        mse = _calculate_mse(target_list, net_outputs)
        acc = _calculate_accuracy(target_list, rounded_net_outputs)

        return mse, acc

    def update_learning_rate(self, momentum):
        if self.epoch_counter % 100 == 0:
            self.learning_rate_init *= momentum
        self.epoch_counter += 1
