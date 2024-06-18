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

| Version: 1.0 | Year: 1987 | Language: BASIC |
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

**Example:**
```
x0     x1     x2     y      Comment
0.001  0.001  0.000  0.002  Addition:        1 + 1 = 2
0.005  0.015  0.000  0.020  Addition:       5 + 15 = 20
0.100  0.010  0.333  0.090  Subtraction:  100 - 10 = 90
0.009  0.001  0.333  0.008  Subtraction:     9 - 1 = 8
0.002  0.002  0.666  0.004  Multiplication:  2 * 2 = 4
0.012  0.003  0.666  0.036  Multiplication: 12 * 3 = 36
0.009  0.003  0.666  0.003  Multiplication:  9 / 3 = 3
0.010  0.002  0.666  0.005  Multiplication: 10 / 2 = 5
```

<br>

| Version: 1.1 | Year: 1988 | Language: 6502 Machine Language |
| :--- | :--- | :--- |

Version 1.1 was a 6502 machine language version of version 1.0. Mostly the same, but offering faster convergence, and could support up to 1024 weights (parameters). 

<br>

| Version: 2.0 | Year: 1992 | Language: C (Borland Turbo C) |
| :--- | :--- | :--- |

Complete rewrite in C, targeting an 8086 XT with 1MB RAM. This version could support up to 128,000 weights (parameters), and was successfully used to both predict stock market prices and classify technical features of stock market price data.

The market prediction model I employed at the time, used a classification model built with version 2.0, to enrich historical price data features with additional technical features, that fed into a regression model also created with version 2.0, to improve prediction accuracy.

<br>

| Version: 1.2 | Year: 1993 | Language: 6502 Machine Language |
| :--- | :--- | :--- |

Fun "retro-computing" project, to upgrade my original Commodore VIC20 code to support a Commodore 64, after being donated a Commodore 64 from my girlfriend at the time.

This version supported up to 16,000 weights. I tested it with simulated stock market data. The results showed that theoretically, an ANN implemented on a Commodore 64 could, in theory, be used for practical stock market prediction and classification tasks in the 1980s, had it been built in the 1980s.

<br>

| Version: 1.3 | Year: 1993 | Language: 6502 Machine Language |
| :--- | :--- | :--- |

Another fun "retro-computing" project to upgrade the Commodore 64 version to use Backpropagation. After getting this to work, I discovered that the Monte Carlo (MC) method I had employed in the original version was actually faster, and better at finding solutions closer to a global optima.                                 

<br>

| Version: 2.1 | Year: 1993 | Language: C, 80x86 Assembly Language |
| :--- | :--- | :--- |

Upgrade to 2.0, which replaced slow C functions with inline 80x86 Assembler versions of those functions. Everything else was the same as version 2.0, but with faster convergence.



