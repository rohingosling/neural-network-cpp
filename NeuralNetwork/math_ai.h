#ifndef MATH_ANN_H
#define MATH_ANN_H

#include <vector>

using namespace std;

class MathAI
{
    public:

    // Data types. 

    typedef vector <double>                    Vector;
    typedef vector <vector <double>>           Matrix;
    typedef vector <vector < vector <double>>> Tensor;

    // Functions. 

    static double Linear               ( double x );
    static double LinearDerivative     ( double x );
    static double Linear               ( double x, double a );
    static double LinearDerivative     ( double x, double a );
    static double Sigmoid              ( double x );
    static double sigmoid              ( double x, double k );
    static double SigmoidDerivative    ( double x );   
    static double SigmoidDerivative    ( double x, double k );
    static double ReLU                 ( double x );
    static double ReLUDerivative       ( double x );
    static double Tanh                 ( double x );
    static double TanhDerivative       ( double x );
    static double Sinusoidal           ( double x );
    static double SinusoidalDerivative ( double x );
    static double sech                 ( double x );
    static double Sgn                  ( double x );
    static double Step                 ( double x );
    static double VectorSum            ( const Vector& v );
    static Vector HadamardProduct      ( const Vector& u, const Vector& v );
    static double GaussianDistribution ();
};

#endif // MATH_ANN_H
