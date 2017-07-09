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
#ifndef __TESTEQUATION__
#define __TESTEQUATION__

#include <mml/equation.h>
#include <mml/numeric.h>
#include <mml/test.h>
#include <mml/vec.h>
#include <stdexcept>

double g1(const mml::vector<double, 3> &x)
{
    return x[0] * x[0] + 2.0 * x[1] * x[1] + 2.0 * x[2] * x[2] + 15;
}

bool test_equation()
{
    bool out = true;

    // Backward Hessian
    {
        // Create equation array
        mml::equation<double, 3, mml::backward> eqs[1] = {g1};

        // Test solving for the local minimum of f1
        mml::vector<double, 3> x0(10.0);
        mml::vector<double, 3> x1;

        // Test min_fast
        double convergence = eqs[0].min_fast(x0, x1, 20, 1E-7);

        // evaluate g1 at x1
        double y1 = g1(x1);

        // Test if found min at (0.0, 0.0, 0.0) at starting point (10.0, 10.0, 10.0)
        out = out && compare(0.0, convergence, 1E-4);
        out = out && compare(15.0, y1, 1E-4);
        out = out && compare(0.0, x1[0], 1E-4);
        out = out && compare(0.0, x1[1], 1E-4);
        out = out && compare(0.0, x1[2], 1E-4);
        if (!out)
        {
            throw std::runtime_error("Failed equation backward min_fast convex");
        }

        // Test min
        convergence = eqs[0].min(x0, x1, 20, 1E-4);

        // evaluate g1 at x1
        y1 = g1(x1);

        // Test if found min at (0.0, 0.0, 0.0) at starting point (10.0, 10.0, 10.0)
        out = out && compare(0.0, convergence, 1E-4);
        out = out && compare(15.0, y1, 1E-4);
        out = out && compare(0.0, x1[0], 1E-4);
        out = out && compare(0.0, x1[1], 1E-4);
        out = out && compare(0.0, x1[2], 1E-4);
        if (!out)
        {
            throw std::runtime_error("Failed equation backward min");
        }

        // Test hessian calculation
        mml::matrix<double, 3, 3> h = mml::backward<double, 3>::hessian(eqs[0], x0, 1E-3);
        out = out && compare(2.0, h.get(0, 0), 1E-4);
        out = out && compare(0.0, h.get(0, 1), 1E-4);
        out = out && compare(0.0, h.get(0, 2), 1E-4);
        out = out && compare(0.0, h.get(1, 0), 1E-4);
        out = out && compare(4.0, h.get(1, 1), 1E-4);
        out = out && compare(0.0, h.get(1, 2), 1E-4);
        out = out && compare(0.0, h.get(2, 0), 1E-4);
        out = out && compare(0.0, h.get(2, 1), 1E-4);
        out = out && compare(4.0, h.get(2, 2), 1E-4);
        if (!out)
        {
            throw std::runtime_error("Failed equation backward hessian");
        }
    }

    // Center Hessian
    { // Create equation array
        mml::equation<double, 3, mml::center> eqs[1] = {g1};

        // Test solving for the local minimum of f1
        mml::vector<double, 3> x0(10.0);
        mml::vector<double, 3> x1;

        // Test min_fast
        double convergence = eqs[0].min_fast(x0, x1, 20, 1E-7);

        // evaluate g1 at x1
        double y1 = g1(x1);

        // Test if found min at (0.0, 0.0, 0.0) at starting point (10.0, 10.0, 10.0)
        out = out && compare(0.0, convergence, 1E-4);
        out = out && compare(15.0, y1, 1E-4);
        out = out && compare(0.0, x1[0], 1E-4);
        out = out && compare(0.0, x1[1], 1E-4);
        out = out && compare(0.0, x1[2], 1E-4);
        if (!out)
        {
            throw std::runtime_error("Failed equation center min_fast convex");
        }

        // Test min
        convergence = eqs[0].min(x0, x1, 20, 1E-4);

        // evaluate g1 at x1
        y1 = g1(x1);

        // Test if found min at (0.0, 0.0, 0.0) at starting point (10.0, 10.0, 10.0)
        out = out && compare(0.0, convergence, 1E-4);
        out = out && compare(15.0, y1, 1E-4);
        out = out && compare(0.0, x1[0], 1E-4);
        out = out && compare(0.0, x1[1], 1E-4);
        out = out && compare(0.0, x1[2], 1E-4);
        if (!out)
        {
            throw std::runtime_error("Failed equation center min");
        }

        // Test hessian calculation
        mml::matrix<double, 3, 3> h = mml::center<double, 3>::hessian(eqs[0], x0, 1E-3);
        out = out && compare(2.0, h.get(0, 0), 1E-4);
        out = out && compare(0.0, h.get(0, 1), 1E-4);
        out = out && compare(0.0, h.get(0, 2), 1E-4);
        out = out && compare(0.0, h.get(1, 0), 1E-4);
        out = out && compare(4.0, h.get(1, 1), 1E-4);
        out = out && compare(0.0, h.get(1, 2), 1E-4);
        out = out && compare(0.0, h.get(2, 0), 1E-4);
        out = out && compare(0.0, h.get(2, 1), 1E-4);
        out = out && compare(4.0, h.get(2, 2), 1E-4);
        if (!out)
        {
            throw std::runtime_error("Failed equation center hessian");
        }
    }

    // Forward Hessian
    {
        // Create equation array
        mml::equation<double, 3, mml::forward> eqs[1] = {g1};

        // Test solving for the local minimum of f1
        mml::vector<double, 3> x0(10.0);
        mml::vector<double, 3> x1;

        // Test min_fast
        double convergence = eqs[0].min_fast(x0, x1, 20, 1E-7);

        // evaluate g1 at x1
        double y1 = g1(x1);

        // Test if found min at (0.0, 0.0, 0.0) at starting point (10.0, 10.0, 10.0)
        out = out && compare(0.0, convergence, 1E-4);
        out = out && compare(15.0, y1, 1E-4);
        out = out && compare(0.0, x1[0], 1E-4);
        out = out && compare(0.0, x1[1], 1E-4);
        out = out && compare(0.0, x1[2], 1E-4);
        if (!out)
        {
            throw std::runtime_error("Failed equation forward min_fast convex");
        }

        // Test min
        convergence = eqs[0].min(x0, x1, 20, 1E-4);

        // evaluate g1 at x1
        y1 = g1(x1);

        // Test if found min at (0.0, 0.0, 0.0) at starting point (10.0, 10.0, 10.0)
        out = out && compare(0.0, convergence, 1E-4);
        out = out && compare(15.0, y1, 1E-4);
        out = out && compare(0.0, x1[0], 1E-4);
        out = out && compare(0.0, x1[1], 1E-4);
        out = out && compare(0.0, x1[2], 1E-4);
        if (!out)
        {
            throw std::runtime_error("Failed equation forward min");
        }

        // Test hessian calculation
        mml::matrix<double, 3, 3> h = mml::forward<double, 3>::hessian(eqs[0], x0, 1E-3);
        out = out && compare(2.0, h.get(0, 0), 1E-4);
        out = out && compare(0.0, h.get(0, 1), 1E-4);
        out = out && compare(0.0, h.get(0, 2), 1E-4);
        out = out && compare(0.0, h.get(1, 0), 1E-4);
        out = out && compare(4.0, h.get(1, 1), 1E-4);
        out = out && compare(0.0, h.get(1, 2), 1E-4);
        out = out && compare(0.0, h.get(2, 0), 1E-4);
        out = out && compare(0.0, h.get(2, 1), 1E-4);
        out = out && compare(4.0, h.get(2, 2), 1E-4);
        if (!out)
        {
            throw std::runtime_error("Failed equation forward hessian");
        }
    }

    return out;
}

#endif