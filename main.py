import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)

        # Set synaptic weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        We train the model through trial and error, adjusting the
        synaptic weights each time to get a better result
        """
        for iteration in range(training_iterations):
            # Pass training set through the neural network
            output = self.think(training_inputs)

            # Calculate the error rate
            error = training_outputs - output

            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def train_with_animation(self, training_inputs, training_outputs,
                             training_iterations, sample_interval=50):
        """
        Train the neural network while displaying a real-time animated graph
        showing synaptic weights, predicted outputs, and mean squared error.

        Parameters
        ----------
        training_inputs : ndarray
        training_outputs : ndarray
        training_iterations : int
            Total number of gradient-descent iterations.
        sample_interval : int
            Record & render a frame every N iterations.
            For 10,000 iterations at interval 50 → 200 frames.
        """
        num_samples = training_iterations // sample_interval

        # ---- history buffers ----
        iterations_recorded = []
        w1_hist, w2_hist, w3_hist = [], [], []
        out0_hist, out1_hist, out2_hist, out3_hist = [], [], [], []
        mse_hist = []

        # ---- setup figure ----
        plt.ion()
        fig, (ax_w, ax_o, ax_e) = plt.subplots(3, 1, figsize=(11, 9))
        fig.canvas.manager.set_window_title('Neural Network — Live Training')

        # -- subplot 1: synaptic weights --
        ax_w.set_title('Synaptic Weights', fontweight='bold')
        ax_w.set_ylabel('Weight value')
        ax_w.set_xlim(0, training_iterations)
        ax_w.set_ylim(-6, 12)
        ax_w.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax_w.grid(True, alpha=0.3)
        (line_w1,) = ax_w.plot([], [], 'C0', label='w₁  (Input 1)')
        (line_w2,) = ax_w.plot([], [], 'C1', label='w₂  (Input 2)')
        (line_w3,) = ax_w.plot([], [], 'C2', label='w₃  (Input 3 / bias)')
        ax_w.legend(loc='upper left', fontsize=8)

        # -- subplot 2: predicted outputs vs targets --
        ax_o.set_title('Predicted Outputs vs Targets', fontweight='bold')
        ax_o.set_ylabel('Output  (sigmoid)')
        ax_o.set_xlim(0, training_iterations)
        ax_o.set_ylim(-0.05, 1.15)
        ax_o.grid(True, alpha=0.3)
        (line_o0,) = ax_o.plot([], [], 'C0', label='Out [0,0,1]')
        (line_o1,) = ax_o.plot([], [], 'C1', label='Out [1,1,1]')
        (line_o2,) = ax_o.plot([], [], 'C2', label='Out [1,0,1]')
        (line_o3,) = ax_o.plot([], [], 'C3', label='Out [0,1,1]')
        # dashed target lines
        targets = training_outputs.flatten()
        colors = ['C0', 'C1', 'C2', 'C3']
        for i, (t, c) in enumerate(zip(targets, colors)):
            ax_o.axhline(y=t, color=c, linewidth=0.8, linestyle='--', alpha=0.5)
        ax_o.legend(loc='center right', fontsize=8, ncol=2)

        # -- subplot 3: mean squared error --
        ax_e.set_title('Mean Squared Error', fontweight='bold')
        ax_e.set_xlabel('Iteration')
        ax_e.set_ylabel('MSE')
        ax_e.set_xlim(0, training_iterations)
        ax_e.set_ylim(-0.01, 0.35)
        ax_e.grid(True, alpha=0.3)
        (line_mse,) = ax_e.plot([], [], 'C3', linewidth=1.5)

        fig.tight_layout()

        # ---- training loop with recording ----
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(
                training_inputs.T,
                error * self.sigmoid_derivative(output),
            )
            self.synaptic_weights += adjustments

            # record frame
            if iteration % sample_interval == 0 or iteration == training_iterations - 1:
                iterations_recorded.append(iteration)
                w = self.synaptic_weights.flatten()
                o = output.flatten()
                w1_hist.append(w[0])
                w2_hist.append(w[1])
                w3_hist.append(w[2])
                out0_hist.append(o[0])
                out1_hist.append(o[1])
                out2_hist.append(o[2])
                out3_hist.append(o[3])
                mse_hist.append(np.mean(error ** 2))

                # update plot data
                line_w1.set_data(iterations_recorded, w1_hist)
                line_w2.set_data(iterations_recorded, w2_hist)
                line_w3.set_data(iterations_recorded, w3_hist)
                line_o0.set_data(iterations_recorded, out0_hist)
                line_o1.set_data(iterations_recorded, out1_hist)
                line_o2.set_data(iterations_recorded, out2_hist)
                line_o3.set_data(iterations_recorded, out3_hist)
                line_mse.set_data(iterations_recorded, mse_hist)

                plt.pause(0.001)  # let the GUI breathe

        # ---- finalize ----
        plt.ioff()
        plt.show(block=False)  # non-blocking so the interactive prompt still works

    def think(self, inputs):
        """
        Pass inputs through the neural network to get output
        """
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    # Initialize the single neuron neural network
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set, with 4 examples consisting of 3
    # input values and 1 output value
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    # Train the neural network with live animation
    neural_network.train_with_animation(training_inputs, training_outputs, 10000, sample_interval=50)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    
    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))
