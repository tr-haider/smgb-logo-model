import numpy as np
#Relu Activation function  => Turns -ve to 0 and postive to remain unchanged
def relu(x):
    return np.maximum(0,x);
data = np.array([-1,0,1,2,-3])
print(relu(data))

#Backpropagtion => Algorithm that implements gradient descent which is used to minimize loss
#Forward prop => Evaluates the loss for each iteration
#Gradient descent used to minimize loss by iteratively adjusting the weights and biases
#Learning rate => It tells gradient descent algorithm to how strongly adjust the wight and biases
#Batch size => Set of examples/input for each iteration to be processed
#Epoch => Ratio between the total dataset size to batch size.Number of training iteration