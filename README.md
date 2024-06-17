Description:

  General purpose neural network class, supporting both regression and classification models.

Usage:

  Regression Problems :

  - Hidden Layers:

    - The most common activation function for hidden layers in regression problems is the Rectified Linear 
      Unit ( ReLU ).// 
    - ReLU allows gradients to flow back efficiently during training ( avoiding vanishing gradients ) and can 
      model non-linear relationships well.
    - Other options include Leaky ReLU, which addresses the "dying ReLU" problem, or even parametric ReLU ( PReLU ) 
      for added flexibility.

  - Output Layer:

    - In regression, the output layer typically uses a linear activation function.
    - This ensures the final output represents a continuous value on the real number line, which aligns with the 
      desired outcome in regression tasks ( e.g., predicting house prices or stock values ).

  Classification Problems :

  - Hidden Layers:

    - Similar to regression, ReLU is a popular choice for hidden layers in classification problems due to its 
      efficiency in training.However, other options like tanh ( squashes values between - 1 and 1 ) or sigmoid
      ( outputs between 0 and 1 ) can also be used.These functions can be helpful if your data naturally falls
      within a specific range.

    - Output Layer:

      - The choice for the output layer depends on the number of classes you're predicting. 

      - Binary Classification ( Two Classes ):
        - Use the sigmoid function to transform the final output into a probability between 0 ( class 1 ) 
          and 1 ( class 2 ).

      - Multi - Class Classification ( More Than Two Classes ):
        - Here, you typically employ the softmax function.
        - Softmax normalizes the output layer's activations into probabilities that sum to 1, representing the
          probability of each class.

  - Note: 

    - The "optimal" activation function can vary depending on your specific dataset and problem.

    - Experimentation is often key.It's a good practice to start with the recommended choices above and then compare
      different options using techniques like grid search or random search to find the best performing combination
      for your situation.


Version History:

  Ordered by year. 

  ------- ------ --------------- --------------------------------------------------------------------
  Version Year   Language        Description
  ------- ------ --------------- --------------------------------------------------------------------

  1.0     1987   BASIC           First attempt at building an ANN, on a Commodore VIC20. A variation
                                 of the Monte Carlo (MC) method was used to to train the weights.

                                 Note:
                                 - I did not know how to implement Backpropagation at the time, hence
                                   the use of MC.
                                 - It later turned out, by accident really, that MC offered faster
                                   convergence, when I later learned how to implement Backpropagation
                                   and compared it with MC.
                               
                                 This initial VIC20 version of the ANN was able to suport up to 256
                                 weights (paramters). While not very practical, it was enough to 
                                 test the ANN on a learning task to learn how to compute binary 
                                 mathmatical operators on 16-bit numbers. The training data was
                                 structured as follows.
                               
                                 Features:
                                   x0 : 16-bit number. Operator operand A.
                                   x1 : 16-bit number. Operator operand B.
                                   x2 : 16-bit number. Operator, (+,-,*,/).
                                                       +(A,B) = 0.000
                                                       -(A,B) = 0.333
                                                       *(A,B) = 0.666
                                                       /(A,B) = 1.000                                
                                 Target/s:
                                   y  : 16-bit number. Normalised result of `y = Operator (A,B)`.
                               
  1.1     1988   6502 ML         Version 1.1 was a 6502 machine language version of version 1.0.
                                 Mostly the same, but offering faster convergence, and could support
                                 up to 1024 weights (paramters). 

  2.0     1992   C (ISO C90)     Complete rewrite in C, targeting an 8086 XT wtih 1MB RAM. This
                                 version could support up to 128,000 weights (paramters), and was
                                 successfuly used to both predict stock market prices, and classify
                                 technical features of stock market price data.
  
                                 The market prediction model I employed at the time, used a 
                                 classification model built with version 2.0, to enrich historical 
                                 price data feeatures with aditional technical features, that fed 
                                 into a regression model also created with version 2.0, to improve 
                                 prediction accuracy.

  1.2     1993   6502 ML         Fun "retro computing" project, to upgrade my original Commodore 
                                 VIC20 code to support a Commodore 64, after being donated a 
                                 Commodore 64 from my girlfriend at the time.

                                 This version supported up to 16,000 weights. I tested it with 
                                 simulated stock market data. The results showed that theoretically
                                 an ANN implemented on a Commodore 64 could, in theory, be used for
                                 practical stock market prediction and classificationin tasks in the
                                 1980s, had it been built in the 1980s.   

  1.3     1993   6502 ML         Another fun "retro computing" project, to upgrade the Commodore 64
                                 version to use Backpropagation. After getting this to work, I 
                                 discovered that the Monte Carlo (MC) method I had emplpyed in the
                                 original version was actualy faster, and bettter at finding 
                                 solutions closer to a global optima.                                     

  2.1     1993   C (ISO C90)     Upgrade to 2.0, that replaced slow C functions with inline 80x86 
                 80x86 Assembly  Assembler versions of those functions. Everything else was the same
                                 as verion 2.0, but with faster convergence.

  2.2     1993   C (ISO C90)     Added support for additional activation functions, loss functions,
                 80x86 Assembly  and optimization functions.
                                 
                                 Supported Activation Functions:
                                 - Linear
                                 - Sigmoid
                                 - TanH

                                 Supported LossFunctions:
                                 - MSE (Mean Squared Error)
                                 - CE  (Cross Entropy)

                                 Supported Optimization Algorythms:
                                 - Backpropagation
                                 - Monte Carlo
                                 - Simulated Annealing

  3.0     1994   C++ (C++2.1)    Complete OOP rewrite in C++, targeting an AMD 80486 DOS machine with
                 80x86 Assembly  2GB of RAM. Code was written using Borland Turbo C++ 3.0.

                                 Same functionality and features as version 2.2, but rewritten in C++
                                 and upgrded memory management with support for up to 500 million
                                 weights (paramters).
                                 
                                 I used version 3.0 for two applications. 
                                 - Stock market regresssion and classificaiton models.
                                 - Language model for implementing experimental chatbots.
                                   I used software programming tutorials as English corpus traning
                                   data, since my goal was to build chat bots capable of generating
                                   C++ and assembly source code. 

  3.1     1995   C++ (C++2.1)    Upgraded version designed to work with a suite of adtional machine
                 80x86 Assembly  learning classes including `RecurrentNeuralNetwork` which was a 
                                 class that implemented an RNN.

                                 Applications:

                                 - Bug simulator, simulating bugs in a fluid that learned how to hunt
                                   and avoid being eaten by other bugs.

                                 - Upgraded stock market prediction model, used in conjunction with 
                                   the `RecurrentNeuralNetwork` class to add better time series 
                                   modeling accuracy.

                                 - Upgradedd chat bot, used in conjuction with the 
                                   `RecurrentNeuralNetwork` class to create a time series based 
                                   language model.

  3.2,3,4 2000   C++ (C++2.1)    Series of incremental impovements from 1996 to 2000 based on new
                 80x86 Assembly  learnings while sudying (BSc computer science). 

  3.5     2001   C++ (C++98)     Major rewrite of all machine learnign classes, to port all code 
                                 written using Borland Turbo C++, to Borland C++ Builder.

                                 Applications:

                                 - Started focusing on Forex market regression and classification 
                                   models.

                                 - Upgraded language model for chatbots, focusing on learning First 
                                   Order Logic (FOL) patterns to improve reasoning.                                    

  4.0     2008   C++ (C++98)     Complete rewtie to support training using paralell computing with    
                 80x86 Assembly  a GPU. Used assemply again after a long pause from assembly, to 
                                 access my Nvidia GeForce instruction set.

                                 Applications:

                                 - Mostly ongoing language model research focusing on training logic
                                   and reasoning.

  4.1     2011   C++ (C++03)     Upgrade to use Nvidia CUDA API.

                                 Note:
                                 Work on my C++ machine learning suite of classes winding down, as 
                                 most new work and research is migrating to Python.

  5.0     2013   C++ (C++11)     Complete rewrite from scratch to support new framework for 
                                 integrating existing ML classes to work with C# .NET trading bots
                                 written for the cTrader cAlgo Forex trading platform.

                                 Status:
                                 - Incomplete.
                                 - Started migrating to Python for all new machine learning 
                                   integration with cTrader for building market prediction and
                                   classification models.
                                 - All language model research migrated to Python using scikit-learn.

[ 5.1 ]   2014   C++ (C++11)     Minor tweeks to support legacy language model and chatbot projects.

                                 Status:
                                 - Current C++ version.
                                 - Incomplete. 

  6.0-FX  2017   Python          Complete rewrite using Keras and scikit-learn.
                                 Specialised for financial time series prediction.
                                 
                                 Applications:

                                 - Numerai financial time-series machine learning competition.
                                   Numerai models built using this version regularily finished in the
                                   top 50. Best position was 35th.

                                 - Forex trading models integrated with C#.NET cAlgo (cTrader)
                                   trading bots. 

  6.0-NLP 2018   Python          Language model specialised version using OpenAI GPT-1.

                                 Applications:

                                 - Used to replace all my own legacy language models with GPT-1
                                   based variations. Bittersweet project, in that all my own 
                                   personal language model research was now overshadowed by 
                                   transformer based models like OpenAI GPT.

  6.1-NLP 2019   Python          Updated to support OpenAI GPT-2.

                                 Applications:

                                 - RAG (Retreval Augmented Generation) enabled chatbot research
                                   projects. 

  ------- ------ --------------- --------------------------------------------------------------------

To-Do:

1. Add support for different types of weight initialisation.

   - Xavier/Glorot Initialization:
     This method works well with activation functions like tanh and sigmoid.
     
   - He Initialization:
     This method works well with activation functions like ReLU.

2. Change the propagate methods to to PropagateForward and PropagateBackward.

3. Add a log file. 


