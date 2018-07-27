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
#ifndef __TESTSYSTEM__
#define __TESTSYSTEM__

#include <mml/equation.h>
#include <mml/numeric.h>
#include <mml/system.h>
#include <mml/test.h>
#include <mml/vec.h>

double f1(const mml::vector<double, 3> &x)
{
    return x[0] + 2.0 * x[1] - 2.0 * x[2] + 15;
}

double f2(const mml::vector<double, 3> &x)
{
    return 2.0 * x[0] + x[1] - 5.0 * x[2] + 21;
}

double f3(const mml::vector<double, 3> &x)
{
    return x[0] - 4.0 * x[1] + x[2] - 18;
}

bool test_system()
{
    bool out = true;

    // Backward gradient test
    {
        // Create equation array
        mml::equation<double, 3, mml::backward> eqs[3] = {f1, f2, f3};

        // Create system of equations
        mml::system<double, 3, mml::backward> system(eqs);

        // Set the x value starting position
        double values[3] = {-1.0, -4.0, 3.0};
        mml::vector<double, 3> x(values);

        // Test system evaluate, should be (0.0, 0.0, 0.0)
        mml::vector<double, 3> y = system.evaluate(x);
        out = out && test(0.0, y[0], 1E-4, "Failed matrix backward identity");
        out = out && test(0.0, y[1], 1E-4, "Failed matrix backward identity");
        out = out && test(0.0, y[2], 1E-4, "Failed matrix backward identity");

        mml::matrix<double, 3, 3> j = system.jacobian(y, 0.1);
        out = out && test(1.0, j.get(0, 0), 1E-4, "Failed matrix backward jacobian");
        out = out && test(2.0, j.get(0, 1), 1E-4, "Failed matrix backward jacobian");
        out = out && test(-2.0, j.get(0, 2), 1E-4, "Failed matrix backward jacobian");
        out = out && test(2.0, j.get(1, 0), 1E-4, "Failed matrix backward jacobian");
        out = out && test(1.0, j.get(1, 1), 1E-4, "Failed matrix backward jacobian");
        out = out && test(-5.0, j.get(1, 2), 1E-4, "Failed matrix backward jacobian");
        out = out && test(1.0, j.get(2, 0), 1E-4, "Failed matrix backward jacobian");
        out = out && test(-4.0, j.get(2, 1), 1E-4, "Failed matrix backward jacobian");
        out = out && test(1.0, j.get(2, 2), 1E-4, "Failed matrix backward jacobian");

        // Test solving the root of system of equations
        mml::vector<double, 3> x0(10.0);
        mml::vector<double, 3> x1;

        // Test zero
        double convergence = system.zero(x0, x1);
        out = out && test(0.0, convergence, 1E-7, "Failed matrix backward zero");
        out = out && test(-1.0, x1[0], 1E-4, "Failed matrix backward zero");
        out = out && test(-4.0, x1[1], 1E-4, "Failed matrix backward zero");
        out = out && test(3.0, x1[2], 1E-4, "Failed matrix backward zero");
    }

    // Center gradient test
    {
        // Create equation array
        mml::equation<double, 3, mml::center> eqs[3] = {f1, f2, f3};

        // Create system of equations
        mml::system<double, 3, mml::center> system(eqs);

        // Set the x value starting position
        double values[3] = {-1.0, -4.0, 3.0};
        mml::vector<double, 3> x(values);

        // Test system evaluate, should be (0.0, 0.0, 0.0)
        mml::vector<double, 3> y = system.evaluate(x);
        out = out && test(0.0, y[0], 1E-4, "Failed matrix center identity");
        out = out && test(0.0, y[1], 1E-4, "Failed matrix center identity");
        out = out && test(0.0, y[2], 1E-4, "Failed matrix center identity");

        mml::matrix<double, 3, 3> j = system.jacobian(y, 0.1);
        out = out && test(1.0, j.get(0, 0), 1E-4, "Failed matrix center jacobian");
        out = out && test(2.0, j.get(0, 1), 1E-4, "Failed matrix center jacobian");
        out = out && test(-2.0, j.get(0, 2), 1E-4, "Failed matrix center jacobian");
        out = out && test(2.0, j.get(1, 0), 1E-4, "Failed matrix center jacobian");
        out = out && test(1.0, j.get(1, 1), 1E-4, "Failed matrix center jacobian");
        out = out && test(-5.0, j.get(1, 2), 1E-4, "Failed matrix center jacobian");
        out = out && test(1.0, j.get(2, 0), 1E-4, "Failed matrix center jacobian");
        out = out && test(-4.0, j.get(2, 1), 1E-4, "Failed matrix center jacobian");
        out = out && test(1.0, j.get(2, 2), 1E-4, "Failed matrix center jacobian");

        // Test solving the root of system of equations
        mml::vector<double, 3> x0(10.0);
        mml::vector<double, 3> x1;

        // Test zero
        double convergence = system.zero(x0, x1);
        out = out && test(0.0, convergence, 1E-7, "Failed matrix center zero");
        out = out && test(-1.0, x1[0], 1E-4, "Failed matrix center zero");
        out = out && test(-4.0, x1[1], 1E-4, "Failed matrix center zero");
        out = out && test(3.0, x1[2], 1E-4, "Failed matrix center zero");
    }

    // Forward gradient test
    {
        // Create equation array
        mml::equation<double, 3, mml::forward> eqs[3] = {f1, f2, f3};

        // Create system of equations
        mml::system<double, 3, mml::forward> system(eqs);

        // Set the x value starting position
        double values[3] = {-1.0, -4.0, 3.0};
        mml::vector<double, 3> x(values);

        // Test system evaluate, should be (0.0, 0.0, 0.0)
        mml::vector<double, 3> y = system.evaluate(x);
        out = out && test(0.0, y[0], 1E-4, "Failed matrix forward identity");
        out = out && test(0.0, y[1], 1E-4, "Failed matrix forward identity");
        out = out && test(0.0, y[2], 1E-4, "Failed matrix forward identity");

        mml::matrix<double, 3, 3> j = system.jacobian(y, 0.1);
        out = out && test(1.0, j.get(0, 0), 1E-4, "Failed matrix forward jacobian");
        out = out && test(2.0, j.get(0, 1), 1E-4, "Failed matrix forward jacobian");
        out = out && test(-2.0, j.get(0, 2), 1E-4, "Failed matrix forward jacobian");
        out = out && test(2.0, j.get(1, 0), 1E-4, "Failed matrix forward jacobian");
        out = out && test(1.0, j.get(1, 1), 1E-4, "Failed matrix forward jacobian");
        out = out && test(-5.0, j.get(1, 2), 1E-4, "Failed matrix forward jacobian");
        out = out && test(1.0, j.get(2, 0), 1E-4, "Failed matrix forward jacobian");
        out = out && test(-4.0, j.get(2, 1), 1E-4, "Failed matrix forward jacobian");
        out = out && test(1.0, j.get(2, 2), 1E-4, "Failed matrix forward jacobian");

        // Test solving the root of system of equations
        mml::vector<double, 3> x0(10.0);
        mml::vector<double, 3> x1;

        // Test zero
        double convergence = system.zero(x0, x1);
        out = out && test(0.0, convergence, 1E-7, "Failed matrix forward zero");
        out = out && test(-1.0, x1[0], 1E-4, "Failed matrix forward zero");
        out = out && test(-4.0, x1[1], 1E-4, "Failed matrix forward zero");
        out = out && test(3.0, x1[2], 1E-4, "Failed matrix forward zero");
    }

    return out;
}

#endif
