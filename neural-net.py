import numpy as np

class NeuralNet():
     
    def __init__(self):
        #find random
        np.random.seed(1)
        array_size = 9
        num_weights = array_size;
        num_dimensions = 4;
        #convert weights converts weights to weights * dimensions with 
        #rando weights[double] (-1,1) keeping mean 0
        self.weights = 2 * np.random.random((num_weights, num_dimensions)) - 1
         
         
         
         
         
    def sigmoid(self, x):
        ##sigmoid func defauild to 
        func = 1 / (1 + np.exp(-x))
        return func 
         
         
         
         
         
    def sigmoid_derivative(self, x):
        ##derivative of default sig func
        derivative = x * (1 - x)
        return derivative
         
         
         
         
         
    def train(self, training_inputs, training_outputs, training_iterations):
        #train model continually
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - outputs
             
             
            #weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            
            self.weights += adjustments
            
             
    def think(self, inputs):
        #passes input via neuron for outputs
        #converts values to floats
         
        input = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_wieights))
        return outputs
         
         
     
if __name__ == "__main__":

    #initializing neuron class
    neural_network = NeuralNet()
    print("beggining randoml generated weights:")
    print(neural_network.weights)
    
    #train data 
    #whateveer we so choose in the form of array of dimension of data
    
    training_inputs = np.array([ [0, 0, 1],
                                 [0, 1, 0],
                                 [0, 1, 1],
                                 [1, 0, 0],
                                 [1, 0, 1],
                                 [1, 1, 0],
                                 [1, 1, 1],
                                 [0, 0, 0],])
    
    #whateveer we so choose in the form of array of dimension of data#
    training_outputs = np.array([[0, 1, 0, 0],
                                 [1, 0, 0, 1],
                                 [0, 1, 1, 1],
                                 [0, 0, 0, 0]]).T
 
	
    print("Ending Weights after training")
    print("neural_network.synaptic_weights")

    user_input_one = int(str(input("User Input One: ")))
    user_input_two = int(str(input("User Input Two: ")))
    user_input_three = int(str(input("User Input Three: ")))
	
    print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
	
    print("New Output Data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    print("!!! WOWAWEEEEWAAAA !!!")
   
    
    
    
    
     

         
         
         
         
         
         
         
         
         
         


