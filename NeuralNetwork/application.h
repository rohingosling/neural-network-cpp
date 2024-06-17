#ifndef APPLICATION_H
#define APPLICATION_H

#include <string>
#include <vector>

#include "neural_network.h"
#include "math_ai.h"

using namespace std;

class Application
{
public:

    // Constructor. 

    Application ();

    // Public methods. 

    void           Run               ();
    string         ToString          () const;
    MathAI::Matrix LoadTrainingData  ( const string& filename, bool ignore_header_row );
    MathAI::Matrix SplitTable        ( const MathAI::Matrix& matrix, long column_index_first, long column_index_last );
    void           GenerateTestData  ( const string& file_name, long row_count );
    void           TestTerminal      ( NeuralNetwork& neural_network );

    // Public constants. 

    const string TERMINAL_MESSAGE_APPLICATION = "[Application] ";
    const string TERMINAL_MESSAGE_SYSTEM      = "[SYSTEM] ";
    const string TERMINAL_MESSAGE_EXCEPTION   = "[EXCEPTION] ";
    const string TERMINAL_MESSAGE_ERROR       = "[ERROR] ";

private:

    // Private constants. 

    string application_name;
    string application_version;
};

#endif // APPLICATION_H