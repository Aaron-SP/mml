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
#ifndef __TESTMATRIX__
#define __TESTMATRIX__

#include <mml/mat.h>
#include <mml/test.h>
#include <mml/vec.h>
#include <stdexcept>

bool test_matrix()
{
    bool out = true;

    // Test matrix operations
    mml::matrix<double, 2, 2> m1;
    mml::matrix<double, 2, 2> m2;
    mml::matrix<double, 3, 3> m3;
    mml::matrix<double, 4, 4> m4;

    // Test identity matrix
    out = out && compare(1.0, m1.get(0, 0), 1E-4);
    out = out && compare(0.0, m1.get(0, 1), 1E-4);
    out = out && compare(0.0, m1.get(1, 0), 1E-4);
    out = out && compare(1.0, m1.get(1, 1), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed matrix identity");
    }

    // Set m2 for add
    m2.get(0, 0) = 1.0;
    m2.get(0, 1) = 2.0;
    m2.get(1, 0) = 3.0;
    m2.get(1, 1) = 4.0;

    // test add
    m1 += m2 + m2;
    out = out && compare(3.0, m1.get(0, 0), 1E-4);
    out = out && compare(4.0, m1.get(0, 1), 1E-4);
    out = out && compare(6.0, m1.get(1, 0), 1E-4);
    out = out && compare(9.0, m1.get(1, 1), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed matrix add");
    }

    // Set m2 for sub
    m2.get(0, 0) = 4.0;
    m2.get(0, 1) = 3.0;
    m2.get(1, 0) = 2.0;
    m2.get(1, 1) = 1.0;

    // test sub
    m1 -= m1 - m2; // -1.0, 1.0, 4.0, 8.0
    out = out && compare(4.0, m1.get(0, 0), 1E-4);
    out = out && compare(3.0, m1.get(0, 1), 1E-4);
    out = out && compare(2.0, m1.get(1, 0), 1E-4);
    out = out && compare(1.0, m1.get(1, 1), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed matrix sub");
    }

    // Set m3 for det
    m3.get(0, 0) = 1.0;
    m3.get(0, 1) = 2.0;
    m3.get(0, 2) = 3.0;
    m3.get(1, 0) = 0.0;
    m3.get(1, 1) = -4.0;
    m3.get(1, 2) = 1.0;
    m3.get(2, 0) = 0.0;
    m3.get(2, 1) = 3.0;
    m3.get(2, 2) = -1.0;

    // Test 3x3 determinant
    out = out && compare(1.0, m3.determinant(), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed matrix determinant 1");
    }

    // Set m3 for det
    m3.get(0, 0) = 5.0;
    m3.get(0, 1) = -2.0;
    m3.get(0, 2) = 1.0;
    m3.get(1, 0) = 0.0;
    m3.get(1, 1) = 3.0;
    m3.get(1, 2) = -1.0;
    m3.get(2, 0) = 2.0;
    m3.get(2, 1) = 0.0;
    m3.get(2, 2) = 7.0;

    // Test 3x3 determinant
    out = out && compare(103.0, m3.determinant(), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed matrix determinant 2");
    }

    // Set m3 for inverse
    m3.get(0, 0) = 3.0;
    m3.get(0, 1) = 0.0;
    m3.get(0, 2) = 2.0;
    m3.get(1, 0) = 2.0;
    m3.get(1, 1) = 0.0;
    m3.get(1, 2) = -2.0;
    m3.get(2, 0) = 0.0;
    m3.get(2, 1) = 1.0;
    m3.get(2, 2) = 1.0;

    // test matrix inverse
    mml::matrix<double, 3, 3> inverse3 = m3.inverse();

    // Test 3x3 determinant
    out = out && compare(0.2, inverse3.get(0, 0), 1E-4);
    out = out && compare(0.2, inverse3.get(0, 1), 1E-4);
    out = out && compare(0.0, inverse3.get(0, 2), 1E-4);
    out = out && compare(-0.2, inverse3.get(1, 0), 1E-4);
    out = out && compare(0.3, inverse3.get(1, 1), 1E-4);
    out = out && compare(1.0, inverse3.get(1, 2), 1E-4);
    out = out && compare(0.2, inverse3.get(2, 0), 1E-4);
    out = out && compare(-0.3, inverse3.get(2, 1), 1E-4);
    out = out && compare(0.0, inverse3.get(2, 2), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed matrix inverse 3x3");
    }

    // Set m4 for inverse
    m4.get(0, 0) = 4.0;
    m4.get(0, 1) = 0.0;
    m4.get(0, 2) = 0.0;
    m4.get(0, 3) = 0.0;
    m4.get(1, 0) = 0.0;
    m4.get(1, 1) = 0.0;
    m4.get(1, 2) = 2.0;
    m4.get(1, 3) = 0.0;
    m4.get(2, 0) = 0.0;
    m4.get(2, 1) = 1.0;
    m4.get(2, 2) = 2.0;
    m4.get(2, 3) = 0.0;
    m4.get(3, 0) = 1.0;
    m4.get(3, 1) = 0.0;
    m4.get(3, 2) = 0.0;
    m4.get(3, 3) = 1.0;

    // test matrix inverse
    mml::matrix<double, 4, 4> inverse4 = m4.inverse();

    // Test 3x3 determinant
    out = out && compare(0.25, inverse4.get(0, 0), 1E-4);
    out = out && compare(0.00, inverse4.get(0, 1), 1E-4);
    out = out && compare(0.00, inverse4.get(0, 2), 1E-4);
    out = out && compare(0.00, inverse4.get(0, 3), 1E-4);
    out = out && compare(0.00, inverse4.get(1, 0), 1E-4);
    out = out && compare(-1.00, inverse4.get(1, 1), 1E-4);
    out = out && compare(1.00, inverse4.get(1, 2), 1E-4);
    out = out && compare(0.00, inverse4.get(1, 3), 1E-4);
    out = out && compare(0.00, inverse4.get(2, 0), 1E-4);
    out = out && compare(0.50, inverse4.get(2, 1), 1E-4);
    out = out && compare(0.00, inverse4.get(2, 2), 1E-4);
    out = out && compare(0.00, inverse4.get(2, 3), 1E-4);
    out = out && compare(-0.25, inverse4.get(3, 0), 1E-4);
    out = out && compare(0.00, inverse4.get(3, 1), 1E-4);
    out = out && compare(0.00, inverse4.get(3, 2), 1E-4);
    out = out && compare(1.00, inverse4.get(3, 3), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed matrix inverse 4x4");
    }

    // set m3 for ludecomposition test
    m3.get(0, 0) = 3.0;
    m3.get(0, 1) = -0.1;
    m3.get(0, 2) = -0.2;
    m3.get(1, 0) = 0.1;
    m3.get(1, 1) = 7.0;
    m3.get(1, 2) = -0.3;
    m3.get(2, 0) = 0.3;
    m3.get(2, 1) = -0.2;
    m3.get(2, 2) = 10.0;

    // set v3 for ludecomposition test
    mml::vector<double, 3> v3;
    v3[0] = 7.85;
    v3[1] = -19.3;
    v3[2] = 71.4;

    // test ludecomp
    v3 = m3.ludecomp(v3);
    out = out && compare(3.00, v3[0], 1E-4);
    out = out && compare(-2.5, v3[1], 1E-4);
    out = out && compare(7.00, v3[2], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed matrix ludecomp");
    }

    return out;
}

#endif