//--------------------------------------------------------------------------------------------------------------------
// Project:     Common Artificial Intelligence Library
// Class:       NeuralNetwork
// Version:     5.1
// Date:        2014-08-02
// Author:      Rohin Gosling
// 
// 
// Description:
//
//   General purpose neural network class, supporting both regression and classification models.
// 
// Usage:
// 
//   Regression Problems :
// 
//   - Hidden Layers:
// 
//     - The most common activation function for hidden layers in regression problems is the Rectified Linear 
//       Unit ( ReLU ). 
//     - ReLU allows gradients to flow back efficiently during training ( avoiding vanishing gradients ) and can 
//       model non-linear relationships well.
//     - Other options include Leaky ReLU, which addresses the "dying ReLU" problem, or even parametric ReLU ( PReLU ) 
//       for added flexibility.
// 
//   - Output Layer:
// 
//     - In regression, the output layer typically uses a linear activation function.
//     - This ensures the final output represents a continuous value on the real number line, which aligns with the 
//       desired outcome in regression tasks ( e.g., predicting house prices or stock values ).
// 
//   Classification Problems :
// 
//   - Hidden Layers:
// 
//     - Similar to regression, ReLU is a popular choice for hidden layers in classification problems due to its 
//       efficiency in training. However, other options like tanh ( squashes values between - 1 and 1 ) or sigmoid
//       ( outputs between 0 and 1 ) can also be used. These functions can be helpful if your data naturally falls
//       within a specific range.
// 
//     - Output Layer:
// 
//       - The choice for the output layer depends on the number of classes you're predicting. 
// 
//       - Binary Classification ( Two Classes ):
//         - Use the sigmoid function to transform the final output into a probability between 0 ( class 1 ) 
//           and 1 ( class 2 ).
// 
//       - Multi-Class Classification ( More Than Two Classes ):
//         - Here, you typically employ the softmax function.
//         - Softmax normalizes the output layer's activations into probabilities that sum to 1, representing the
//           probability of each class.
// 
//   - Note: 
// 
//     - The "optimal" activation function can vary depending on your specific dataset and problem.
// 
//     - Experimentation is often key. It's a good practice to start with the recommended choices above and then compare
//       different options using techniques like grid search or random search to find the best-performing combination
//       for your situation.
// 
// 
// Version History:
// 
//   Ordered by year. 
// 
//   ------- ------ --------------- --------------------------------------------------------------------
//   Version Year   Language        Description
//   ------- ------ --------------- --------------------------------------------------------------------
// 
//   1.0     1987   BASIC           First attempt at building an ANN, on a Commodore VIC20. A variation
//                                  of the Monte Carlo (MC) method was used to train the weights.
// 
//                                  Note:
//                                  - I did not know how to implement Backpropagation at the time, hence
//                                    the use of MC.
//                                  - It later turned out, by accident really, that MC offered faster
//                                    convergence, when I later learned how to implement Backpropagation
//                                    and compared it with MC.
//                                
//                                  This initial VIC20 version of the ANN was able to support up to 256
//                                  weights (parameters). While not very practical, it was enough to 
//                                  test the ANN on a learning task to learn how to compute binary 
//                                  mathematical operators on 16-bit numbers. The training data was
//                                  structured as follows.
//                                
//                                  Features:
//                                    x0 : 16-bit number. Operator operand A.
//                                    x1 : 16-bit number. Operator operand B.
//                                    x2 : 16-bit number. Operator, (+,-,*,/).
//                                                        +(A,B) = 0.000
//                                                        -(A,B) = 0.333
//                                                        *(A,B) = 0.666
//                                                        /(A,B) = 1.000                                
//                                  Target/s:
//                                    y  : 16-bit number. Normalized result of `y = Operator (A,B)`.
//                                
//   1.1     1988   6502 ML         Version 1.1 was a 6502 machine language version of version 1.0.
//                                  Mostly the same, but offering faster convergence, and could support
//                                  up to 1024 weights (parameters). 
// 
//   2.0     1992   C (ISO C90)     Complete rewrite in C, targeting an 8086 XT with 1MB RAM. This
//                                  version could support up to 128,000 weights (parameters), and was
//                                  successfully used to both predict stock market prices and classify
//                                  technical features of stock market price data.
//   
//                                  The market prediction model I employed at the time, used a 
//                                  classification model built with version 2.0, to enrich historical 
//                                  price data features with additional technical features, that fed 
//                                  into a regression model also created with version 2.0, to improve 
//                                  prediction accuracy.
// 
//   1.2     1993   6502 ML         Fun "retro-computing" project, to upgrade my original Commodore 
//                                  VIC20 code to support a Commodore 64, after being donated a 
//                                  Commodore 64 from my girlfriend at the time.
// 
//                                  This version supported up to 16,000 weights. I tested it with 
//                                  simulated stock market data. The results showed that theoretically
//                                  an ANN implemented on a Commodore 64 could, in theory, be used for
//                                  practical stock market prediction and classification tasks in the
//                                  1980s, had it been built in the 1980s.   
// 
//   1.3     1993   6502 ML         Another fun "retro-computing" project, to upgrade the Commodore 64
//                                  version to use Backpropagation. After getting this to work, I 
//                                  discovered that the Monte Carlo (MC) method I had employed in the
//                                  original version was actually faster, and better at finding 
//                                  solutions closer to global optima.                                     
// 
//   2.1     1993   C (ISO C90)     Upgrade to 2.0, which replaced slow C functions with inline 80x86 
//                  80x86 Assembly  Assembler versions of those functions. Everything else was the same
//                                  as version 2.0, but with faster convergence.
// 
//   2.2     1993   C (ISO C90)     Added support for additional activation functions, loss functions,
//                  80x86 Assembly  and optimization functions.
//                                  
//                                  Supported Activation Functions :
//                                  -Linear
//                                  - Sigmoid
//                                  - TanH
//                                  
//                                  Supported LossFunctions :
//                                  -MSE ( Mean Squared Error )
//                                  - CE ( Cross-Entropy )
//                                  
//                                  Supported Optimization Algorithms :
//                                  -Backpropagation
//                                  - Monte Carlo
//                                  - Simulated Annealing
// 
//   3.0     1994   C++ (C++2.1)    Complete OOP rewrite in C++, targeting an AMD 80486 DOS machine with
//                  80x86 Assembly  2GB of RAM. Code was written using Borland Turbo C++ 3.0.
// 
//                                  Same functionality and features as version 2.2, but rewritten in C++
//                                  and upgraded memory management with support for up to 500 million
//                                  weights (parameters).
//                                  
//                                  I used version 3.0 for two applications. 
//                                  - Stock market regression and classification models.
//                                  - Language model for implementing experimental chatbots.
//                                    I used software programming tutorials as English corpus training
//                                    data, since my goal was to build chatbots capable of generating
//                                    C++ and assembly source code. 
// 
//   3.1     1995   C++ (C++2.1)    Upgraded version designed to work with a suite of additional machine
//                  80x86 Assembly  learning classes including `RecurrentNeuralNetwork` which was a 
//                                  class that implemented an RNN.
// 
//                                  Applications:
// 
//                                  - Bug simulator, simulating bugs in a fluid that learned how to hunt
//                                    and avoid being eaten by other bugs.
// 
//                                  - Upgraded stock market prediction model, used in conjunction with 
//                                    the `RecurrentNeuralNetwork` class to add better time series 
//                                    modeling accuracy.
// 
//                                  - Upgraded chatbot, used in conjunction with the 
//                                    `RecurrentNeuralNetwork` class to create a time series based 
//                                    language model.
// 
//   3.2,3,4 2000   C++ (C++2.1)    Series of incremental improvements from 1996 to 2000 based on new
//                  80x86 Assembly  learnings while studying (BSc computer science). 
// 
//   3.5     2001   C++ (C++98)     Major rewrite of all machine learning classes, to port all code 
//                                  written using Borland Turbo C++, to Borland C++ Builder.
// 
//                                  Applications:
// 
//                                  - Started focusing on Forex market regression and classification 
//                                    models.
// 
//                                  - Upgraded language model for chatbots, focusing on learning First 
//                                    Order Logic (FOL) patterns to improve reasoning.                                    
// 
//   4.0     2008   C++ (C++98)     Complete rewrite to support training using parallel computing with    
//                  80x86 Assembly  a GPU. Used assembly again after a long pause from assembly, to 
//                                  access my Nvidia GeForce instruction set.
// 
//                                  Applications:
// 
//                                  - Mostly ongoing language model research focusing on training logic
//                                    and reasoning.
// 
//   4.1     2011   C++ (C++03)     Upgrade to use Nvidia CUDA API.
// 
//                                  Note:
//                                  Work on my C++ machine learning suite of classes winding down, as 
//                                  most new work and research is migrating to Python.
// 
//   5.0     2013   C++ (C++11)     Complete rewrite from scratch to support new framework for 
//Integrating existing ML classes to work with C# .NET trading bots
//                                  written for the cTrader cAlgo Forex trading platform.
// 
//                                  Status:
//                                  - Incomplete.
//                                  - Started migrating to Python for all new machine learning 
//                                    integration with cTrader for building market prediction and
//                                    classification models.
//                                  - All language model research migrated to Python using scikit-learn.
// 
// [ 5.1 ]   2014   C++ (C++11)     Minor tweaks to support legacy language model and chatbot projects.
// 
//                                  Status:
//                                  - Current C++ version.
//                                  - Incomplete. 
// 
//   6.0-FX  2017   Python          Complete rewrite using Keras and scikit-learn.
//                                  Specialised in financial time series prediction.
//                                  
//                                  Applications:
// 
//                                  - Numerai financial time-series machine learning competition.
//                                    Numerai models built using this version regularly finished in the
//                                    top 50. Best position was 35th.
// 
//                                  - Forex trading models integrated with C#.NET cAlgo (cTrader)
//                                    trading bots. 
// 
//   6.0-NLP 2018   Python          Language model specialized version using OpenAI GPT-1.
// 
//                                  Applications:
// 
//                                  - Used to replace all my own legacy language models with GPT-1
//                                    based variations. Bittersweet project, in that all my own 
//                                    personal language model research was now overshadowed by 
//                                    transformer-based models like OpenAI GPT.
// 
//   6.1-NLP 2019   Python          Updated to support OpenAI GPT-2.
// 
//                                  Applications:
// 
//                                  - RAG (Retrieval Augmented Generation) enabled chatbot research
//                                    projects. 
// 
//   ------- ------ --------------- --------------------------------------------------------------------
// 
// To-Do:
// 
// 1. Add support for different types of weight initialization.
// 
//    - Xavier/Glorot Initialization:
//      This method works well with activation functions like tanh and sigmoid.
//      
//    - He Initialization:
//      This method works well with activation functions like ReLU.
// 
// 2. Change the propagate methods to PropagateForward and PropagateBackward.
// 
// 3. Add a log file. 
// 
// 
//--------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include "neural_network.h"
#include "math_ai.h"

using namespace std;

//--------------------------------------------------------------------------------------------------------------------
// Constructor. 
//--------------------------------------------------------------------------------------------------------------------

NeuralNetwork::NeuralNetwork
(
    const vector <int>&                layers,
    const vector <ActivationFunction>& activation_functions,
    LossFunction                       loss_function,
    double                             learning_rate,
    int                                epoch_count,
    int                                batch_size,
    OptimizationAlgorithm              optimization_algorithm,
    const string                       training_results_file_name
) : m_layers                     ( layers ), 
    m_activation_functions       ( activation_functions ),
    m_loss_function              ( loss_function ),
    m_learning_rate              ( learning_rate ),
    m_epoch_count                ( epoch_count ),
    m_batch_size                 ( batch_size ),
    m_optimization_algorithm     ( optimization_algorithm ),
    m_training_results_file_name ( training_results_file_name )
{
    InitializeWeights ();
}

//--------------------------------------------------------------------------------------------------------------------
// 
// Initialize the weights and biases of the neural network.
// 
// Method Name:
// 
// - InitializeWeights 
// 
// 
// Description:
// 
// - Initializes the weights and biases for each layer of the neural network using a random uniform distribution.
// 
// - The method performs the following steps:
//   1. Iterates over each layer of the network except the output layer.
//   2. For each layer, initializes a weight vector and a bias vector with random values from a uniform distribution.
//   3. Adds the initialized weight and bias vectors to the corresponding class members (`m_weight_vectors` and 
//      `m_bias_vectors`).
// 
// - Note:
//   - The weights and biases are initialized to random values to break symmetry during training.
//   - The random values are drawn from a uniform distribution in the range [-1.0, 1.0].
// 
// 
// Parameters:
// 
// - None
// 
// 
// Return Value:
// 
// - None
// 
// 
// Preconditions:
// 
// - The network architecture (`m_layers`) must be properly defined and initialized.
// 
// 
// Postconditions:
// 
// - The weights and biases for each layer are initialized with random values.
// 
// - The `m_weight_vectors` and `m_bias_vectors` members are populated with the initialized values.
// 
// 
// Functional Decomposition:
// 
// - InitializeWeights
// 
// 
// To-Do:
// 
// 1. Add support for different types of weight initialization methods (e.g., Xavier, He).
// 
// 2. Add parameterization to allow customization of the initialization range.
// 
// 3. Optimize the initialization process for better performance.
// 
//--------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::InitializeWeights ()
{
    default_random_engine              generator;
    uniform_real_distribution <double> distribution ( -1.0, 1.0 );

    for ( size_t i = 0; i < m_layers.size () - 1; ++i )
    {
        int layer_size      = m_layers [ i ];
        int layer_size_next = m_layers [ i + 1 ];

        MathAI::Vector layer_weight_vector ( layer_size_next * layer_size );
        MathAI::Vector layer_bias_vector   ( layer_size_next );

        for ( double& weight : layer_weight_vector ) weight = distribution ( generator );
        for ( double& bias   : layer_bias_vector   ) bias   = distribution ( generator );

        m_weight_vectors.push_back ( layer_weight_vector );
        m_bias_vectors.push_back   ( layer_bias_vector   );
    }
}

//--------------------------------------------------------------------------------------------------------------------
// 
// Train the neural network using backpropagation.
//
// Method Name:
// 
// - Train 
// 
// 
// Description:
// 
// - Trains the neural network using the provided training data.
// 
// - The method performs the following steps:
//   1. Initializes variables for the number of samples, batch size, and epochs.
//   2. Opens a CSV file to log the training results.
//   3. Iterates over each epoch and within each epoch, iterates over batches.
//   4. Initializes and accumulates gradients for each batch.
//   5. Performs forward and backward passes for each sample in the batch.
//   6. Calculates and accumulates gradients for weights and biases.
//   7. Updates weights and biases using the accumulated gradients.
//   8. Computes the loss for each epoch and logs it to the CSV file and console.
//   9. Closes the CSV file after training is complete.
// 
// - Note:
//   - The method supports batch training and updates the weights and biases after processing each batch.
//   - The method computes the average loss over all samples to monitor the training progress.
// 
// 
// Parameters:
// 
// - training_data_x : Tensor2D
//   The input features of the training data. Each row represents a sample, and each column represents a feature.
// 
// - training_data_y : Tensor2D
//   The target values of the training data. Each row represents a sample, and each column represents a target value.
// 
// 
// Return Value:
// 
// - None
// 
// 
// Preconditions:
// 
// - The training data must be preprocessed and normalized if required.
// 
// - The input and target data must have matching numbers of samples (rows).
// 
// - The network architecture and hyperparameters must be properly initialized.
// 
// 
// Postconditions:
// 
// - The weights and biases of the neural network are updated based on the training data.
// 
// - The network is trained for the specified number of epochs.
// 
// - The training loss is logged to a CSV file and optionally printed to the console for monitoring.
// 
// 
// Functional Decomposition:
// 
// - Train
//   - InitializeGradients
//   - ForwardPropagate
//   - BackwardPropagate
//   - AccumulateGradients
//   - UpdateWeightsAndBiases
//   - ComputeLoss
// 
// 
// To-Do:
// 
// 1. Implement additional optimization algorithms such as Adam.
// 
// 2. Implement early stopping to prevent overfitting.
// 
// 3. Add support for learning rate scheduling.
// 
// 4. Add validation data support to monitor overfitting during training.
// 
// 5. Check for optimal use of `size_t` vs `int` vs `long`.
// 
// 6. Clean up training result file management. 
// 
//--------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::Train ( const MathAI::Matrix& training_data_x, const MathAI::Matrix& training_data_y )
{
    ShowCursor ( false );

    // Initialise local variables.

    size_t epoch_count         = m_epoch_count;                             // Number of training epochs.
    size_t batch_sample_count  = m_batch_size;                              // Number of samples per batch.    
    size_t global_sample_count = training_data_x.size ();                   // Total number of samples in the global training dataset.
    size_t batch_count         = global_sample_count / batch_sample_count;  // Compute the number of batches per epoch.

    // Open the CSV file for writing training results and add CSV header row to the file. 

    ofstream results_file ( m_training_results_file_name );

    if ( !results_file.is_open () )
    {
        cerr << "Error: Unable to open training results file: " << m_training_results_file_name << endl;
        return;
    }    

    results_file << "epoch_index,loss\n";

    // Train the neural network over the specified number of epochs.

    for ( size_t epoch_index = 0; epoch_index < epoch_count; ++epoch_index )
    {
        // Itterate over epoch training batches. 

        for ( size_t batch_index = 0; batch_index < batch_count; ++batch_index )
        {
            MathAI::Matrix weight_gradients ( m_weight_vectors.size () );     // Initialize weight gradients for this batch.
            MathAI::Matrix bias_gradients   ( m_bias_vectors.size   () );     // Initialize bias gradients for this batch.

            InitializeGradients ( weight_gradients, bias_gradients );         // Set gradients to zero.

            // Iterate over each sample in the current training batch.

            for ( int sample_index = 0; sample_index < batch_sample_count; ++sample_index )
            {
                size_t   batch_sample_index    = batch_index * batch_sample_count + sample_index;                              // Calculate the global sample index.
                MathAI::Matrix activations     = PropagateForward  ( training_data_x [ batch_sample_index ] );                 // Perform forward pass.
                MathAI::Matrix error_gradients = PropagateBackward ( activations, training_data_y [ batch_sample_index ] );    // Perform backward pass.

                AccumulateGradients ( activations, error_gradients, weight_gradients, bias_gradients );                        // Accumulate gradients.
            }

            UpdateWeightsAndBiases ( weight_gradients, bias_gradients );  // Update weights and biases using accumulated gradients.
        }

        // Compute loss.
        // - Write epoch loss to results file. 
        // - Write epoch loss to terminal file. 

        double loss = ComputeLoss ( training_data_x, training_data_y );
        results_file << epoch_index << "," << loss << "\n";
        cout << TrainingProgressToString ( epoch_index, epoch_count, loss ) << flush;
    }

    // Clean up and exit training.
    // - Close training results file. 
    // - Enable the terminal cursor and write a new line to the terminal. 

    results_file.close ();
    ShowCursor ( true );
    cout << endl;
}

//--------------------------------------------------------------------------------------------------------------------
// 
// Initialize gradients for weights and biases for a training batch.
// 
// Method Name:
// 
// - InitializeGradients 
// 
// 
// Description:
// 
// - Initializes the weight and bias gradient vectors to zero for each layer of the neural network.
// 
// - The method performs the following steps:
//   1. Iterates over each layer of the network.
//   2. Resizes the weight gradient vector to match the size of the corresponding weight vector and sets all elements to 0.0.
//   3. Resizes the bias gradient vector to match the size of the corresponding bias vector and sets all elements to 0.0.
// 
// - Note:
//   - This method is called at the beginning of processing each batch during training to reset the gradients.
// 
// 
// Parameters:
// 
// - weight_gradients : Tensor2D
//   The weight gradients to be initialized. Each element is a vector representing the gradients of a layer's weights.
// 
// - bias_gradients : Tensor2D
//   The bias gradients to be initialized. Each element is a vector representing the gradients of a layer's biases.
// 
// 
// Return Value:
// 
// - None
// 
// 
// Preconditions:
// 
// - The weight and bias gradient vectors must have the same size as the weight and bias vectors of the network.
// 
// 
// Postconditions:
// 
// - The weight and bias gradient vectors are resized to match the corresponding weight and bias vectors and set to 0.0.
// 
// 
// Functional Decomposition:
// 
// - InitializeGradients
// 
// 
// To-Do:
// 
// 1. Optimize gradient initialization for better performance.
// 
// 2. Add support for gradient initialization methods other than setting to zero, if needed for advanced optimization algorithms.
// 
// 3. Ensure compatibility with different network architectures and initialization methods.
// 
//--------------------------------------------------------------------------------------------------------------------


void NeuralNetwork::InitializeGradients ( MathAI::Matrix& weight_gradients, MathAI::Matrix& bias_gradients )
{
    // Initialize gradients for the current training batch.

    for ( size_t layer_index = 0; layer_index < m_weight_vectors.size (); ++layer_index )
    {
        weight_gradients [ layer_index ].resize ( m_weight_vectors [ layer_index ].size (), 0.0 );
        bias_gradients   [ layer_index ].resize ( m_bias_vectors   [ layer_index ].size (), 0.0 );
    }
}

//--------------------------------------------------------------------------------------------------------------------
// 
// Accumulate gradients for weights and biases during backpropagation.
// 
// Method Name:
// 
// - AccumulateGradients 
// 
// 
// Description:
// 
// - Accumulates the weight and bias gradients for each layer of the neural network based on the error gradients.
// 
// - The method performs the following steps:
//   1. Iterates over each layer of the network (except the input layer).
//   2. For each layer, iterates over each neuron to calculate the weight gradients using the error gradients and activations.
//   3. Accumulates the bias gradients using the error gradients.
// 
// - Note:
//   - This method is called during the training process after computing the error gradients to update the weight and bias gradients.
//   - The weight gradients are calculated as the product of the error gradient and the activation of the previous layer's neuron.
// 
// 
// Parameters:
// 
// - activations : Tensor2D
//   The activations of each layer from the forward pass. Each element is a vector representing the activations of a layer.
// 
// - error_gradients : Tensor2D
//   The error gradients for each layer from the backward pass. Each element is a vector representing the gradients of a layer.
// 
// - weight_gradients : Tensor2D
//   The weight gradients to be accumulated. Each element is a vector representing the gradients of a layer's weights.
// 
// - bias_gradients : Tensor2D
//   The bias gradients to be accumulated. Each element is a vector representing the gradients of a layer's biases.
// 
// 
// Return Value:
// 
// - None
// 
// 
// Preconditions:
// 
// - The forward and backward passes must be completed, and the activations and error gradients must be available.
// 
// - The weight and bias gradient vectors must be properly initialized and sized to match the network's architecture.
// 
// 
// Postconditions:
// 
// - The weight and bias gradient vectors are updated based on the error gradients and activations.
// 
// 
// Functional Decomposition:
// 
// - AccumulateGradients
// 
// 
// To-Do:
// 
// 1. Optimize gradient accumulation for better performance.
// 
// 2. Ensure compatibility with different network architectures and activation functions.
// 
// 3. Add support for advanced gradient accumulation techniques such as gradient clipping.
// 
//--------------------------------------------------------------------------------------------------------------------


void NeuralNetwork::AccumulateGradients
(
    const MathAI::Matrix& activations,
    const MathAI::Matrix& error_gradients,
          MathAI::Matrix& weight_gradients,
          MathAI::Matrix& bias_gradients
)
{
    for ( size_t layer_index = 1; layer_index < m_weight_vectors.size (); ++layer_index )
    {
        for ( size_t neuron_index = 0; neuron_index < m_weight_vectors [ layer_index ].size (); ++neuron_index )
        {
            size_t layer_neuron_count_next = m_layers [ layer_index + 1 ];
            size_t activation_index        = neuron_index % layer_neuron_count_next;

            if ( activation_index < activations [ layer_index ].size () )
            {
                double neuron_gradient                           = error_gradients [ layer_index ][ neuron_index ];
                double previous_layer_neuron_activation          = activations [ layer_index ][ activation_index ];
                weight_gradients [ layer_index ][ neuron_index ] = neuron_gradient * previous_layer_neuron_activation;
            }
            else
            {
                cerr << "[Train] ERROR:" << endl
                     << "Index out of bounds : activation_index = " << activation_index                    << endl
                     << "activations[layer_index].size() = "        << activations [ layer_index ].size () << endl;
            }
        }

        for ( size_t neuron = 0; neuron < m_bias_vectors [ layer_index ].size (); ++neuron )
        {
            double neuron_error_gradient              = error_gradients [ layer_index + 1 ][ neuron ];
            bias_gradients [ layer_index ][ neuron ] += neuron_error_gradient;
        }
    }
}

//--------------------------------------------------------------------------------------------------------------------
// 
// Update weights and biases using accumulated gradients.
// 
// Method Name:
// 
// - UpdateWeightsAndBiases 
// 
// 
// Description:
// 
// - Updates the weights and biases for each layer of the neural network using the accumulated gradients.
// 
// - The method performs the following steps:
//   1. Iterates over each layer of the network.
//   2. For each layer, iterates over each neuron to update the weights using the accumulated weight gradients.
//   3. Updates the biases using the accumulated bias gradients.
// 
// - Note:
//   - The weights and biases are updated by subtracting the product of the learning rate and the corresponding gradient
//     divided by the batch size.
//   - This method is called during the training process after accumulating the gradients for each batch.
// 
// 
// Parameters:
// 
// - weight_gradients : Tensor2D
//   The accumulated weight gradients. Each element is a vector representing the gradients of a layer's weights.
// 
// - bias_gradients : Tensor2D
//   The accumulated bias gradients. Each element is a vector representing the gradients of a layer's biases.
// 
// 
// Return Value:
// 
// - None
// 
// 
// Preconditions:
// 
// - The weight and bias gradient vectors must be properly initialized and sized to match the network's architecture.
// 
// - The learning rate and batch size must be properly defined and initialized.
// 
// 
// Postconditions:
// 
// - The weights and biases of the neural network are updated based on the accumulated gradients.
// 
// 
// Functional Decomposition:
// 
// - UpdateWeightsAndBiases
// 
// 
// To-Do:
// 
// 1. Implement support for advanced optimization algorithms such as Adam and RMSprop.
// 
// 2. Optimize the weight and bias update process for better performance.
// 
// 3. Add support for adaptive learning rates.
// 
//--------------------------------------------------------------------------------------------------------------------


void NeuralNetwork::UpdateWeightsAndBiases ( const MathAI::Matrix& weight_gradients, const MathAI::Matrix& bias_gradients )
{
    // Initialise local variables.

    size_t weight_vector_count = m_weight_vectors.size ();

    // update weights and biases.

    for ( size_t layer_index = 0; layer_index < weight_vector_count; ++layer_index )
    {
        size_t layer_weight_vector_count = m_weight_vectors [ layer_index ].size ();
        size_t layer_bias_vector_count   = m_bias_vectors   [ layer_index ].size ();

        for ( size_t neuron_index = 0; neuron_index < layer_weight_vector_count; ++neuron_index )
        {
            double neuron_weight_gradient                     = weight_gradients [ layer_index ][ neuron_index ];
            m_weight_vectors [ layer_index ][ neuron_index ] -= m_learning_rate * neuron_weight_gradient / m_batch_size;
        }

        for ( size_t neuron_index = 0; neuron_index < layer_bias_vector_count; ++neuron_index )
        {
            double neuron_bias_gradient                     = bias_gradients [ layer_index ][ neuron_index ];
            m_bias_vectors [ layer_index ][ neuron_index ] -= m_learning_rate * neuron_bias_gradient / m_batch_size;
        }
    }
}

//--------------------------------------------------------------------------------------------------------------------
// Show or hide the cursor. 
//--------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::ShowCursor ( bool cursor_visible )
{
    if ( cursor_visible )
    {
        cout << "\033[?25h";
    }
    else
    {
        cout << "\033[?25l";
    }

    cout << flush;
}

//--------------------------------------------------------------------------------------------------------------------
// 
// Perform a forward pass through the neural network.
//  
// Method Name:
// 
// - ForwardPropagate 
// 
// 
// Description:
// 
// - Computes the activations for each layer in the neural network given an input sample.
// 
// - The method performs the following steps:
//   1. Initializes the activations vector to store activations for all layers.
//   2. Stores the input activations.
//   3. Iterates over each layer (except the input layer) to compute the activations.
//   4. For each neuron_index in the current layer, computes the weighted sum of activations from the previous layer plus the
//      bias term.
//   5. Applies the activation function to the sum to get the neuron_index activation.
//   6. Stores the activations for the current layer.
//   7. Returns the activations for all layers.
// 
// - Note:
//   - The forward pass is essential for making predictions and calculating the loss during training.
// 
// 
// Parameters:
// 
// - input : Tensor1D
//   The input features for a single sample. A vector representing the input to the neural network.
// 
// 
// Return Value:
// 
// - activations : Tensor2D
//   The activations for each layer. Each element is a vector representing the activations of a layer.
// 
//   Note:
//   - The activations are used in the backward pass to compute the gradients.
// 
// 
// Preconditions:
// 
// - The input size must match the size of the input layer of the network.
// 
// - The network architecture and weights must be properly initialized.
// 
// 
// Postconditions:
// 
// - The activations for each layer are computed and stored.
// 
// - The activations can be used for predictions or for the backward pass during training.
// 
// 
// Functional Decomposition:
// 
// - ForwardPropagate
// 
// 
// To-Do:
// 
// 1. Implement support for different types of activation functions.
// 
// 2. Optimize the forward pass for better performance.
// 
// 3. Add support for dropout during training.
// 
//--------------------------------------------------------------------------------------------------------------------

MathAI::Matrix NeuralNetwork::PropagateForward ( const MathAI::Vector& input )
{
    MathAI::Matrix activations;
    activations.push_back ( input );

    for ( size_t i = 0; i < m_layers.size () - 1; ++i )
    {
        MathAI::Vector layer_output ( m_layers [ i + 1 ] );

        for ( size_t j = 0; j < m_layers [ i + 1 ]; ++j )
        {
            // Compute network function.

            double sum = m_bias_vectors [ i ][ j ];

            for ( size_t k = 0; k < m_layers [ i ]; ++k )
            {
                double x = activations.back () [ k ];
                double w = m_weight_vectors [ i ][ k * m_layers [ i + 1 ] + j ];

                sum += w * x;
            }

            // Select and compute activation function.

            switch ( m_activation_functions [ i ] )
            {
                case LINEAR:  layer_output [ j ] = MathAI::Linear  ( sum ); break;
                case SIGMOID: layer_output [ j ] = MathAI::Sigmoid ( sum ); break;
                case TANH:    layer_output [ j ] = MathAI::Tanh    ( sum ); break;
                case RELU:    layer_output [ j ] = MathAI::ReLU    ( sum ); break;
            }
        }

        activations.push_back ( layer_output );
    }

    return activations;
}

//--------------------------------------------------------------------------------------------------------------------
// 
// Compute the error gradients using backpropagation.
// 
// Method Name:
// 
// - BackwardPropagate 
// 
// 
// Description:
// 
// - Computes the error gradients for each layer of the neural network using backpropagation.
// 
// - The method performs the following steps:
//   1. Initializes the delta vector for each layer.
//   2. Calculates the delta for the output layer based on the error between the predicted and target values.
//   3. Propagates the error backwards through the hidden layers, calculating deltas using the weighted sum of deltas
//      from the next layer and the derivative of the activation function.
// 
// - Note:
//   - The deltas represent the gradients of the loss with respect to the activations for each layer.
// 
// 
// Parameters:
// 
// - activations : Tensor2D
//   The activations of each layer from the forward pass. Each element is a vector representing the activations of a
//   layer.
// 
// - targets : Tensor1D
//   The target values for the current training sample. A vector representing the expected output.
// 
// 
// Return Value:
// 
// - deltas : Tensor2D
//   The error gradients for each layer. Each element is a vector representing the gradients of a layer.
// 
//   Note:
//   - The deltas are used to compute the weight and bias gradients during the training process.
// 
// 
// Preconditions:
// 
// - The forward pass must be completed, and the activations must be available.
// 
// - The target values must match the network's output size.
// 
// 
// Postconditions:
// 
// - The deltas for each layer are computed and returned.
// 
// - The deltas can be used to compute the gradients for weights and biases.
// 
// 
// Functional Decomposition:
// 
// - BackwardPropagate
// 
// 
// To-Do:
// 
// 1. Implement support for different loss functions beyond mean squared error.
// 
// 2. Add support for batch normalization and its gradient computation.
// 
// 3. Optimize the backward pass for better performance.
// 
//--------------------------------------------------------------------------------------------------------------------


MathAI::Matrix NeuralNetwork::PropagateBackward ( const MathAI::Matrix& activations, const MathAI::Vector& targets )
{
    MathAI::Matrix deltas ( m_layers.size () );

    // Calculate output layer delta.

    deltas.back ().resize ( m_layers.back () );

    for ( size_t i = 0; i < m_layers.back (); ++i )
    {
        double error      = activations.back () [ i ] - targets [ i ];
        double activation = activations.back () [ i ];

        switch ( m_activation_functions.back () )
        {
            case LINEAR:  deltas.back () [ i ] = error * MathAI::LinearDerivative  ( activation ); break;
            case SIGMOID: deltas.back () [ i ] = error * MathAI::SigmoidDerivative ( activation ); break;
            case TANH:    deltas.back () [ i ] = error * MathAI::TanhDerivative    ( activation ); break;
            case RELU:    deltas.back () [ i ] = error * MathAI::ReLUDerivative    ( activation ); break;
        }
    }

    // Calculate hidden layer deltas.

    for ( size_t i = m_layers.size () - 2; i > 0; --i )
    {
        deltas [ i ].resize ( m_layers [ i ] );

        for ( size_t j = 0; j < m_layers [ i ]; ++j )
        {
            double activation = activations [ i ][ j ];
            double delta      = 0.0;

            for ( size_t k = 0; k < m_layers [ i + 1 ]; ++k )
            {
                delta += deltas [ i + 1 ][ k ] * m_weight_vectors [ i ][ j * m_layers [ i + 1 ] + k ];
            }

            switch ( m_activation_functions [ i - 1 ] )
            {
                case LINEAR:  deltas [ i ][ j ] = delta * MathAI::LinearDerivative  ( activation ); break;
                case SIGMOID: deltas [ i ][ j ] = delta * MathAI::SigmoidDerivative ( activation ); break;
                case TANH:    deltas [ i ][ j ] = delta * MathAI::TanhDerivative    ( activation ); break;
                case RELU:    deltas [ i ][ j ] = delta * MathAI::ReLUDerivative    ( activation ); break;
            }
        }
    }

    return deltas;
}

//--------------------------------------------------------------------------------------------------------------------
// 
// Compute the loss for the given training data.
// 
// Method Name:
// 
// - ComputeLoss 
// 
// 
// Description:
// 
// - Computes the mean squared error (MSE) loss for the given training data.
// 
// - The method performs the following steps:
//   1. Iterates over each sample in the training data.
//   2. Performs a forward pass to get the network's predictions for each sample.
//   3. Calculates the loss (squared error) for each output neuron_index.
//   4. Accumulates the squared errors for all samples.
//   5. Returns the average loss over all samples.
// 
// - Note:
//   - The loss is used to monitor the training progress and adjust the weights and biases during training.
// 
// 
// Parameters:
// 
// - training_data_x : Tensor2D
//   The input features of the training data. Each row represents a sample, and each column represents a feature.
// 
// - training_data_y : Tensor2D
//   The target values of the training data. Each row represents a sample, and each column represents a target value.
// 
// 
// Return Value:
// 
// - loss : double
//   The mean squared error (MSE) loss for the given training data.
// 
//   Note:
//   - The loss is used to monitor the training progress and adjust the weights and biases during training.
// 
// 
// Preconditions:
// 
// - The training data must be preprocessed and normalized if required.
// 
// - The input and target data must have matching numbers of samples (rows).
// 
// - The network architecture and weights must be properly initialized.
// 
// 
// Postconditions:
// 
// - The mean squared error (MSE) loss for the given training data is computed and returned.
// 
// - The loss can be used to monitor the training progress and adjust the weights and biases during training.
// 
// 
// Functional Decomposition:
// 
// - ComputeLoss
//   - ForwardPropagate
// 
// 
// To-Do:
// 
// 1. Implement support for different loss functions beyond mean squared error.
// 
// 2. Optimize the loss computation for better performance.
// 
// 3. Add support for batch loss computation.
// 
//--------------------------------------------------------------------------------------------------------------------

double NeuralNetwork::ComputeLoss ( const MathAI::Matrix& training_data_x, const MathAI::Matrix& training_data_y )
{
    double training_data_sample_count = (double) training_data_x.size ();
    double loss                       = 0.0;
    double average_loss               = 0.0;

    // Accumulate loss measurements. 

    for ( size_t i = 0; i < training_data_x.size (); ++i )
    {
        MathAI::Matrix        activations = PropagateForward ( training_data_x [ i ] );
        const MathAI::Vector& output      = activations.back ();

        for ( size_t j = 0; j < output.size (); ++j )
        {
            double error = output [ j ] - training_data_y [ i ][ j ];
            loss += error * error;
        }
    }

    // Compute average loss. 

    average_loss = loss / training_data_sample_count;

    return average_loss;
}

//--------------------------------------------------------------------------------------------------------------------
// Regression inference.
//--------------------------------------------------------------------------------------------------------------------

MathAI::Vector NeuralNetwork::Predict ( const MathAI::Vector& input )
{
    // Perform forward propagation on the input data.

    MathAI::Matrix activations = PropagateForward ( input );

    // The output of the network is the activations of the last layer.

    return activations.back ();
}

//--------------------------------------------------------------------------------------------------------------------
// Classification inference. 
//--------------------------------------------------------------------------------------------------------------------

int NeuralNetwork::Classify ( const MathAI::Vector& input )
{
    // Perform forward propagation on the input data.

    MathAI::Matrix activations = PropagateForward ( input );

    // The output of the network is the activations of the last layer.

    std::vector<double> output = activations.back ();

    // Find the index of the maximum value in the output vector.

    auto max_element_it =       max_element ( output.begin (), output.end () );
    int predicted_class = (int) distance    ( output.begin (), max_element_it );

    return predicted_class;
}

//--------------------------------------------------------------------------------------------------------------------
// SaveModel
//--------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::SaveModel ( const string& model_path )
{
    // Stub for saving model.
}

//--------------------------------------------------------------------------------------------------------------------
// LoadModel
//--------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::LoadModel ( const string& model_path )
{
    // Stub for loading model.
}

//--------------------------------------------------------------------------------------------------------------------
// Print training progress. 
//--------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::TrainingProgressToString ( size_t epoch_index, size_t epoch_count, double loss )
{
    // Initialise local constants.

    const string CARIAGE_RETURN               = "\r";
    const string TRAIN                        = "[Train] ";
    const string EPOCH_FIELD                  = "Epoch: ";
    const string OUT_OF                       = " / ";
    const string DELIMITER                    = ", ";
    const string LOSS_FIELD                   = "Loss: ";
    const string PARENTHESIS_PERCENTAGE_OPEN  = " (";
    const string PARENTHESIS_PERCENTAGE_CLOSE = " %)";

    // Initialise local variables. 

    size_t training_progress_percentage        = ( size_t ) ( 100 * ( epoch_index + 1 ) / epoch_count );  // Compute training progress. 
    string epoch_index_string                  = to_string ( epoch_index + 1 );
    string epoch_count_string                  = to_string ( epoch_count );
    string training_progress_percentage_string = to_string ( training_progress_percentage );
    string loss_string                         = to_string ( loss );
    string training_progress_string            = "";

    // Compile progress strings. 

    training_progress_string += TRAIN;
    training_progress_string += EPOCH_FIELD + epoch_index_string + OUT_OF + epoch_count_string;
    training_progress_string += PARENTHESIS_PERCENTAGE_OPEN + training_progress_percentage_string + PARENTHESIS_PERCENTAGE_CLOSE;
    training_progress_string += DELIMITER + LOSS_FIELD + loss_string;
    training_progress_string += CARIAGE_RETURN;

    // Return progress string to caller. 

    return training_progress_string;
}

//--------------------------------------------------------------------------------------------------------------------
// ToString
//--------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::ToString () const
{
    // Local constants. 

    const string NEW_LINE            = "\n";
    const string NA                  = "N/A";
    const string BULLET              = "- ";
    const string DELIMITER_1         = ", ";
    const string DELIMITER_2         = ": ";
    const string ASSIGNMENT          = " = ";    
    const string LAYER_SIZE          = "Neuron Count";
    const string ACTIVATION_FUNCTION = "Activation Function";
    const string LAYER               = "Layer ";
    const string LAYER_INPUT         = "[ Input  ] ";
    const string LAYER_HIDDEN        = "[ Hidden ] ";
    const string LAYER_OUTPUT        = "[ Output ] ";

    // Initialise local variables. 

    string bullet                    = "- ";
    string application_string        = "";
    size_t layer_count               = this->m_layers.size ();
    size_t activation_function_count = this->m_activation_functions.size ();

    // Compile string.    

    application_string += NEW_LINE;
    application_string += "Neural Network Layers:" + NEW_LINE;

    // Compile string: Compile layer info. 

    for ( size_t i = 0; i < layer_count; i++ )
    {
        string layer_info                 = "";        
        string layer_size                 = to_string ( this->m_layers [ i ] );
        string layer_index_string         = to_string ( i );
        string activation_function_string = "";
        string layer_type                 = "";
        size_t activation_function_index = ( ( i - 1 >= 0 ) && ( i - 1 < activation_function_count ) ) ? i - 1 : 0;

        if ( i == 0 )
        {
            layer_type                 = LAYER_INPUT;
            activation_function_string = NA;            
        }
        else if ( i < layer_count - 1 )
        {
            layer_type                 = LAYER_HIDDEN;
            activation_function_string = ActivationFunctionToString ( this->m_activation_functions [ activation_function_index ] );
        }
        else
        {
            layer_type                 = LAYER_OUTPUT;
            activation_function_string = ActivationFunctionToString ( this->m_activation_functions [ activation_function_index ] );
        }

        layer_info += bullet + layer_type + LAYER + layer_index_string + DELIMITER_2;
        layer_info += LAYER_SIZE + ASSIGNMENT + layer_size + DELIMITER_1;
        layer_info += ACTIVATION_FUNCTION + ASSIGNMENT + activation_function_string + NEW_LINE;

        application_string += layer_info;
    }
       

    // Compile scalar meta-parameters. 

    application_string += NEW_LINE;
    application_string += "Loss Function          = " + LossFunctionToString          ( this->m_loss_function          ) + NEW_LINE;
    application_string += "Optimization Algorythm = " + OptimizationAlgorithmToString ( this->m_optimization_algorithm ) + NEW_LINE;
    application_string += "Learning Rate          = " + FormatFloat                   ( this->m_learning_rate          ) + NEW_LINE;
    application_string += "Epoch Count            = " + to_string                     ( this->m_epoch_count            ) + NEW_LINE;
    application_string += "Batch Count            = " + to_string                     ( this->m_batch_size             ) + NEW_LINE;


    return ( application_string );
}

//--------------------------------------------------------------------------------------------------------------------
// ActivationFunctionToString
//--------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::ActivationFunctionToString ( ActivationFunction activation_function ) const
{
    switch ( activation_function )
    {
        case ActivationFunction::LINEAR:  return "LINEAR";
        case ActivationFunction::SIGMOID: return "SIGMOID";
        case ActivationFunction::TANH:    return "TANH";
        case ActivationFunction::RELU:    return "RELU";
        default:                          return "";
    }
}

//--------------------------------------------------------------------------------------------------------------------
// LossFunctionToString
//--------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::LossFunctionToString ( LossFunction loss_function ) const
{
    switch ( loss_function )
    {
        case LossFunction::MEAN_SQUARED_ERROR: return "MEAN_SQUARED_ERROR";
        case LossFunction::CROSS_ENTROPY:      return "CROSS_ENTROPY";
        default:                               return "";
    }
}

//--------------------------------------------------------------------------------------------------------------------
// OptimizationAlgorithmToString
//--------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::OptimizationAlgorithmToString ( OptimizationAlgorithm optimization_algorythm ) const
{
    switch ( optimization_algorythm )
    {
        case OptimizationAlgorithm::GRADIENT_DESCENT:            return "GRADIENT_DESCENT";
        case OptimizationAlgorithm::STOCHASTIC_GRADIENT_DESCENT: return "STOCHASTIC_GRADIENT_DESCENT";
        case OptimizationAlgorithm::ADAM:                        return "ADAM";
        default:                                                 return "";
    }
}

//--------------------------------------------------------------------------------------------------------------------
// FormatFloat
//--------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::FormatFloat ( double value ) const
{
    string str = std::to_string ( value );

    str.erase ( str.find_last_not_of ( '0' ) + 1, string::npos );
    str.erase ( str.find_last_not_of ( '.' ) + 1, string::npos );

    return str;
}


//--------------------------------------------------------------------------------------------------------------------
// 
// Method tagline. A one-sentence catchphrase describing what the method does. 
// 
// Method Name:
// 
// - MethodName 
// 
// 
// Description:
// 
// - A bullet point description of what the method does, how it works, and where it is used else where in the class. 
// 
// - Use numbered lists where necessary to explain sections of the method that involve step-by-step sequences of
//   events. For example.
//   1. Step 1
//   2. Step 2
//   3. Step 3
// 
// - Note:
//   - Add bullet notes in the `Description` section of the comments if the need arises.
//   - Use notes in the `Description` section, in the event that it makes sense to do so.
//   - While on the topic of notes, wrap comment lines when the comment line reaches 120 characters long.
//     - When wrapping, keep the left margins neat. So wrap back to where the current line of text starts to keep
//       paragraphing neat and tidy.
//   - Leave two lines between each section of the comments for readability.
// 
// 
// Parameters:
// 
// - parameter_a : DataType
//   Description of `paramter_a`, and how it is used.
// 
// - parameter_b : DataType
//   Description of `paramter_b`, and how it is used.
// 
// - parameter_c : DataType
//   Description of `paramter_c`, and how it is used.
// 
// 
// Return Value:
// 
// - return_value : DataType
//   Description of `return_value`, and how it is used.
//   Note:
//   - Unless data is returned by reference through method arguments, then all return values must be assigned to a
//     local variable.
//   - Example:
//     ```C++11
//     double SomeMethod ( double x0, double x1 )
//     {
//         double y = x0 + x1;
//         return y;
//     }
//     ```
// 
// 
// Preconditions:
// 
// - Description of the first precondition.
// 
// - Description of the second precondition.
// 
// - Description of the third precondition.
// 
// - And so on.
// 
// 
// Postconditions:
// 
// - Description of the first postcondition.
// 
// - Description of the second postcondition.
// 
// - Description of the third postcondition.
// 
// - And so on.
// 
// 
// Functional Decomposition:
// 
// - MethodName
//   - SomeMethod
//   - SomeOtherMethod
//     - AMethodUsedBySomeOtherMethod
//     - AnotherMethodUSedBySomeOtherMethod
//   - YetAnotherMethod
//     - AMethodUsedByYetAnotherMethod
// 
// - Note:
//   - If the method does not call any other methods, just place the name of the method.
//   - Example:
//     - MethodThatDoesNotCallAnyOtherMethods
// 
// 
// To-Do:
// 
// 1. The most important thing I need to consider doing next to improve the method.
// 
// 2. The second most important thing I need to consider doing next to improve the method.
// 
// 3. The third most important thing I need to consider doing next to improve the method.
// 
// 4. And so on.
// 
//--------------------------------------------------------------------------------------------------------------------


