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
#ifndef __TESTVECTOR__
#define __TESTVECTOR__

#include <mml/test.h>
#include <mml/vec.h>
#include <stdexcept>

bool test_vector()
{
    bool out = true;

    // Test vector operations
    mml::vector<double, 2> v1;
    mml::vector<double, 2> v2;

    // Test identity vector
    out = out && compare(0.0, v1[0], 1E-4);
    out = out && compare(0.0, v1[1], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed vector identity");
    }

    // Test v2 for add
    v2[0] = 1.0;
    v2[1] = 2.0;

    // test add
    v1 += v2 + v2;
    out = out && compare(2.0, v1[0], 1E-4);
    out = out && compare(4.0, v1[1], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed vector add");
    }

    // Test v2 for sub
    v2[0] = 3.0;
    v2[1] = 6.0;

    // test sub
    v1 -= v1 - v2;
    out = out && compare(3.0, v1[0], 1E-4);
    out = out && compare(6.0, v1[1], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed vector sub");
    }

    // Test v2 for mult
    v2[0] = 2.0;
    v2[1] = 4.0;

    // test mult
    v1 *= v1 * v2;
    out = out && compare(18.0, v1[0], 1E-4);
    out = out && compare(144.0, v1[1], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed vector mult");
    }

    // Test v2 for div
    v2[0] = 4.0;
    v2[1] = 2.0;

    // test div
    v1 /= v1 / v2;
    out = out && compare(4.0, v1[0], 1E-4);
    out = out && compare(2.0, v1[1], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed vector div");
    }

    // Test constant mathematical operations
    v1 = (((v2 + 1.0) - 2.0) * 3.0) / 4.0;
    out = out && compare(2.25, v1[0], 1E-4);
    out = out && compare(0.75, v1[1], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed vector constant operations");
    }

    // Test constant mathematical operations
    v2 += 1.0;
    v2 -= 2.0;
    v2 *= 3.0;
    v2 /= 4.0;

    out = out && compare(2.25, v2[0], 1E-4);
    out = out && compare(0.75, v2[1], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed vector constant operations");
    }

    return out;
}

#endif