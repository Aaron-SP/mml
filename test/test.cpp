/* Copyright [2013-2016] [Aaron Springstroh, Minimal Math Library]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <iostream>
#include <mml/tequation.h>
#include <mml/tmat.h>
#include <mml/tmult.h>
#include <mml/tnnet.h>
#include <mml/tsystem.h>
#include <mml/tvec.h>

int main()
{
    try
    {
        bool out = true;
        out = out && test_matrix();
        out = out && test_neural_net();
        out = out && test_vector();
        out = out && test_matrix_multiply();
        out = out && test_equation();
        out = out && test_system();
        if (out)
        {
            std::cout << "Math tests passed!" << std::endl;
            return 0;
        }
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
    }

    std::cout << "Math tests failed!" << std::endl;
    return -1;
}
