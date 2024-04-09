import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)

np.random.shuffle(data)

tst_data = pd.read_csv('test.csv')

tst_data = np.array(tst_data)


def ReLU(Z):
    return np.maximum(Z, 0) #Straightforard ReLU implementation for a matrix

def ReLU_deriv(Z):
    return np.where(Z > 0, 1, 0)

def softmax(Z):                                    #Fast softmax implementation i copied from the internet
    max_values = np.amax(Z, axis=0, keepdims=True)  #Get the maximum values along the rows
    exp_values = np.exp(Z - max_values)            #Subtract maximum values for numerical stability and apply exponentiation
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1    #Calculate the weighted sums and biases for the next layer
    A1 = ReLU(Z1)            #Apply the activation function
    Z2 = W2.dot(A1) + b2    #Reiterate, this time with softmax
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def cost_mse(target, output):
    return np.mean(np.square(output - target)) #define the mean squared error function

def back_prop(Z1, A1, Z2, A2, W1, W2, X, target, samples):
    #Calculate the derivatives for each step, using the chain rule
    Delta2 = A2 - target        
    dW2 = Delta2.dot(A1.T) / samples
    db2 = np.sum(Delta2 * 1) / samples

    Delta1 = W2.T.dot(Delta2) * ReLU_deriv(Z1)
    dW1 = Delta1.dot(X.T) / samples
    db1 = np.sum(Delta1 * 1) / samples

    return dW1, db1, dW2, db2

def make_prediction(W1, b1, W2, b2, X):
        _, _, _, a2 = forward_prop(W1, b1, W2, b2, X)    #prediction function, used to test the model with the unlabeled test set
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
        self.human_test_data = test_data
        self.create_weight_matrices()
        self.initiate_biases()
        self.prepare_data()

    def create_weight_matrices(self):
        self.matrix_one = np.random.rand(self.no_hidden_nodes, self.no_input_nodes) * np.sqrt(2 / self.no_input_nodes)
        self.matrix_two = np.random.rand(self.no_output_nodes, self.no_hidden_nodes) * np.sqrt(2 / self.no_hidden_nodes)

    def initiate_biases(self):
        self.bias_one = 0
        self.bias_two = 0

    def prepare_data(self):  #create an 80/20 split
        test_size = self.input.shape[0] // 5
        self.test_data = self.input[:test_size]
        self.train_data = self.input[test_size + 1:]

    def train(self):
        batched_training_data = [self.train_data[i*self.batchsize:(i+1)*self.batchsize] for i in range(len(self.train_data) // self.batchsize)] #Create batches of data for stochastic gradient descent
        test_targets = self.test_data.T[0]
        test_values = self.test_data.T[1:]
        for sublist in batched_training_data:
            tr_sublist = sublist.T
            targets = tr_sublist[0]
            values = tr_sublist[1:] / 255
            one_hot_matrix = np.zeros((self.no_output_nodes, self.batchsize))
            for i, val in enumerate(targets):
                one_hot_column = np.zeros(self.no_output_nodes)    #Create a one-hot encoded column vector for the current element
                one_hot_column[val] = 1
                one_hot_matrix[:, i] = one_hot_column  #Assign the one-hot column vector to the corresponding column in the matrix
                
            Z1, A1, Z2, A2 = forward_prop(self.matrix_one, self.bias_one, self.matrix_two, self.bias_two, values)

            train_predictions = np.argmax(A2, axis=0)

            training_accuracy = np.sum(train_predictions == targets) / targets.size

            _, _, _, final_activation = forward_prop(self.matrix_one, self.bias_one, self.matrix_two, self.bias_two, test_values)

            testing_predictions = np.argmax(final_activation, axis=0)

            testing_accuracy = np.sum(testing_predictions == test_targets) / test_targets.size

            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, self.matrix_one, self.matrix_two, values, one_hot_matrix, self.batchsize)

            self.matrix_one -= self.learning_rate * dW1    #Apply gradient descent
            self.bias_one -= self.learning_rate * db1
            self.matrix_two -= self.learning_rate * dW2
            self.bias_two -= self.learning_rate * db2

            print(f'Training accuracy: {training_accuracy}\n Testing accuracy: {testing_accuracy}\n')
            

    def test(self):
        data = self.human_test_data
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



Cifrck = NNetwork(784, 256, 10, 0.01, data, 64, tst_data)

Cifrck.train()
