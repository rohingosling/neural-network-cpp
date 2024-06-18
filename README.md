# General Purpose Multi-layer Perceptron Network Class
## Usage:
### Regression Problems :
- **Hidden Layers:**
  - The most common activation function for hidden layers in regression problems is the Rectified Linear 
    Unit ( ReLU ). 
  - ReLU allows gradients to flow back efficiently during training ( avoiding vanishing gradients ) and can 
    model non-linear relationships well.
  - Other options include Leaky ReLU, which addresses the "dying ReLU" problem, or even parametric ReLU ( PReLU ) 
    for added flexibility.
    
- **Output Layer:**
  - In regression, the output layer typically uses a linear activation function.
  - This ensures the final output represents a continuous value on the real number line, which aligns with the 
      desired outcome in regression tasks ( e.g., predicting house prices or stock values ).

### Classification Problems :
- **Hidden Layers:**
  - Similar to regression, ReLU is a popular choice for hidden layers in classification problems due to its 
    efficiency in training.
  - However, other options like tanh ( squashes values between - 1 and 1 ) or sigmoid ( outputs between 0 and 1 )
    can also be used.
  - These functions can be helpful if your data naturally falls
    within a specific range.
    
- **Output Layer:**
  - The choice for the output layer depends on the number of classes you're predicting.
  - **Binary Classification ( Two Classes ):**
    - Use the sigmoid function to transform the final output into a probability between 0 ( class 1 ) 
      and 1 ( class 2 ).
  - **Multi-Class Classification ( More Than Two Classes ):**
    - Here, you typically employ the softmax function.
    - Softmax normalizes the output layer's activations into probabilities that sum to 1, representing the
      probability of each class.      

**Note:**
- The "optimal" activation function can vary depending on your specific dataset and problem.
- Experimentation is often key. It's a good practice to start with the recommended choices above and then compare
  different options using techniques like grid search or random search to find the best-performing combination
  for your situation.

## Version History:
*Ordered by year.*

| Version: 1.0 | Year: 1987 | Language: BASCIC |
| :--- | :--- | :--- |

First attempt at building an ANN, on a Commodore VIC20. A variation of the Monte Carlo (MC) method was used to train the weights.<be>

**Note:**
- I did not know how to implement Backpropagation at the time, hence the use of MC.
- It later turned out, by accident really, that MC offered faster convergence, when I later learned how to implement Backpropagation and compared it with MC.

This initial VIC20 version of the ANN was able to support up to 256 weights (parameters). While not very practical, it was enough to test the ANN on a learning task to learn how to compute binary mathematical operators on 16-bit numbers.

Because the VIC20 had less than 5k of usable RAM, the training data was generated in real-time row-by-row, eliminating the need for a file of training data. The training data was structured as follows.

**Features:**
- x0 : 16-bit number. Operator operand A.
- x1 : 16-bit number. Operator operand B.
- x2 : 16-bit number. Operator, (+,-,*,/).
  - +(A,B) = 0.000
  - -(A,B) = 0.333
  - *(A,B) = 0.666
  - /(A,B) = 1.000

**Target/s:**
- y : 16-bit number. Normalized result of, *y = Operator (A,B)*.


                        
