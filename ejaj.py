import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm

data = pd.read_csv('train.csv')

data = np.array(data)

tst_data = np.genfromtxt('test.csv', delimiter=',', skip_header=1)


def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return np.where(Z > 0, 1, 0)

def softmax(matrix):
    max_values = np.amax(matrix, axis=0, keepdims=True)  # Get the maximum values along the rows
    exp_values = np.exp(matrix - max_values)            # Subtract maximum values for numerical stability and apply exponentiation
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):

    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def target_vector(target):
    vec = np.zeros(10, dtype=int)
    vec[target] = 1
    return vec

def cost_mse(target, output):
    return np.mean(np.square(output - target))

def back_prop(Z1, A1, Z2, A2, W1, W2, X, target, samples):
    
    Delta2 = A2 - target
    dW2 = Delta2.dot(A1.T) / samples
    db2 = np.sum(Delta2 * 1) / samples

    Delta1 = W2.T.dot(Delta2) * ReLU_deriv(Z1)
    dW1 = Delta1.dot(X.T) / samples
    db1 = np.sum(Delta1 * 1) / samples

    return dW1, db1, dW2, db2

def make_prediction(W1, b1, W2, b2, X):
        _, _, _, a2 = forward_prop(W1, b1, W2, b2, X)
        print(f'Prediction: {np.argmax(a2)}')

class NNetwork:
    def __init__(self,
                 no_input_nodes,
                 no_hidden_nodes,
                 no_output_nodes,
                 learning_rate,
                 input_data,
                 batchsize,
                 test_data):
        self.no_input_nodes = no_input_nodes
        self.no_hidden_nodes = no_hidden_nodes
        self.no_output_nodes = no_output_nodes
        self.learning_rate = learning_rate
        self.input = input_data
        self.batchsize = batchsize
        self.test_data = test_data
        self.create_weight_matrices()
        self.initiate_biases()

    def create_weight_matrices(self):
        self.matrix_one = np.random.rand(self.no_hidden_nodes, self.no_input_nodes) * np.sqrt(2 / self.no_input_nodes)
        self.matrix_two = np.random.rand(self.no_output_nodes, self.no_hidden_nodes) * np.sqrt(2 / self.no_hidden_nodes)

    def initiate_biases(self):
        self.bias_one = 0
        self.bias_two = 0


    def train(self):
        data = [self.input[i*self.batchsize:(i+1)*self.batchsize] for i in range(len(self.input) // self.batchsize)]
        for sublist in data:
            tr_sublist = sublist.T
            targets = tr_sublist[0]
            values = tr_sublist[1:] / 255

            one_hot_matrix = np.zeros((self.no_output_nodes, self.batchsize))
            for i, val in enumerate(targets):
                # Create a one-hot encoded column vector for the current element
                one_hot_column = np.zeros(self.no_output_nodes)
                one_hot_column[val] = 1
                
                # Assign the one-hot column vector to the corresponding column in the matrix
                one_hot_matrix[:, i] = one_hot_column
            

            Z1, A1, Z2, A2 = forward_prop(self.matrix_one, self.bias_one, self.matrix_two, self.bias_two, values)

            predictions = np.argmax(A2, axis=0)

            accuracy = np.sum(predictions == targets) / targets.size

            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, self.matrix_one, self.matrix_two, values, one_hot_matrix, self.batchsize)

            self.matrix_one -= self.learning_rate * dW1
            self.bias_one -= self.learning_rate * db1
            self.matrix_two -= self.learning_rate * dW2
            self.bias_two -= self.learning_rate * db2

            print(accuracy)

    def test(self):
        data = self.test_data
        total = 0
        correct = 0
        for index, image in enumerate(data):
            prediction = make_prediction(self.matrix_one, self.bias_one, self.matrix_two, self.bias_two, image)
            slika = image.reshape(28, 28)
            plt.subplot(1, 1, 1)
            plt.imshow(slika, cmap='gray')
            plt.axis('off')  # Hide axes
            plt.title(f"Number {prediction}")
            plt.show()
            answer = input('Correct? ')
            if int(answer) == 1:
                total += 1
                correct += 1
            else:
                total +=1
            print(f"Accuracy: {correct/total}")


    #def get_accuracy(self):
    #    _, _, _, a2 = forward_prop(self.matrix_one, self.bias_one, self.matrix_two, self.bias_two, self.input.T[1:])

Cifrck = NNetwork(784, 256, 10, 0.055, data, 64, tst_data)

Cifrck.train()

Cifrck.test()

#Cifrck.make_prediction()

#print(pd.DataFrame(test_sample.reshape(28,28)))