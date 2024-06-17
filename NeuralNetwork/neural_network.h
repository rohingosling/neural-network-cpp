#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

#include "math_ai.h"

using namespace std;

// Enum definitions.

enum ActivationFunction    { LINEAR, SIGMOID, TANH, RELU };
enum LossFunction          { MEAN_SQUARED_ERROR, CROSS_ENTROPY };
enum OptimizationAlgorithm { GRADIENT_DESCENT, STOCHASTIC_GRADIENT_DESCENT, ADAM };

class NeuralNetwork
{

public:

    // Public data types.

    //typedef vector <double>          Vector;
    //typedef vector <vector <double>> Matrix;

    // Constructor.

    NeuralNetwork
    (
        const vector <int>&                layers,
        const vector <ActivationFunction>& activation_functions,
        LossFunction                       loss_function,
        double                             learning_rate,
        int                                epoch_count,
        int                                batch_size,
        OptimizationAlgorithm              optimization_algorithm,
        const string                       training_results_file_name
    );

    // Public methods. 

    void           Train                         ( const MathAI::Matrix& training_data_x, const MathAI::Matrix& training_data_y );
    MathAI::Vector Predict               ( const MathAI::Vector& input );
    int            Classify           ( const MathAI::Vector& input );
    string         ToString                      () const;
    string         ActivationFunctionToString    ( ActivationFunction    activation_function    ) const;
    string         LossFunctionToString          ( LossFunction          loss_function          ) const;
    string         OptimizationAlgorithmToString ( OptimizationAlgorithm optimization_algorythm ) const;
    void           SaveModel                     ( const string& model_path );
    void           LoadModel                     ( const string& model_path );
    string         FormatFloat                   ( double value ) const;


private:

    // Private methods.

    void           InitializeWeights      ();
    MathAI::Matrix PropagateForward       ( const MathAI::Vector& input );
    MathAI::Matrix PropagateBackward      ( const MathAI::Matrix& activations, const MathAI::Vector& targets );
    double         ComputeLoss            ( const MathAI::Matrix& training_data_x, const MathAI::Matrix& training_data_y );
    void           InitializeGradients    ( MathAI::Matrix& weight_gradients, MathAI::Matrix& bias_gradients );
    void           AccumulateGradients    ( const MathAI::Matrix& activations, const MathAI::Matrix& error_gradients, MathAI::Matrix& weight_gradients, MathAI::Matrix& bias_gradients );
    void           UpdateWeightsAndBiases ( const MathAI::Matrix& weight_gradients, const MathAI::Matrix& bias_gradients );

    // Private utility methods. 

    string TrainingProgressToString ( size_t epoch_index, size_t epoch_count, double loss );
    void   ShowCursor               ( bool cursor_hidden );

    // Private member variables.

    vector <int>                m_layers;
    vector <ActivationFunction> m_activation_functions;
    LossFunction                m_loss_function;
    double                      m_learning_rate;
    int                         m_epoch_count;
    int                         m_batch_size;                   // The number of training samples per batch. 
    OptimizationAlgorithm       m_optimization_algorithm;
    MathAI::Matrix              m_weight_vectors;
    MathAI::Matrix              m_bias_vectors;
    string                      m_training_results_file_name;   // CSV file listing the results of a taining run.
};

#endif // NEURAL_NETWORK_H
