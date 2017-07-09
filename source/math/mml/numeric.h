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
#ifndef __NUMERIC__
#define __NUMERIC__

#include <mml/equation.h>
#include <mml/mat.h>
#include <mml/vec.h>

namespace mml
{

// These classes calculate derivatives with various finite difference methods
// A custom gradient class can be created by a user that returns analytical derivatives of each function

// First order backward finite difference
template <typename T, size_t N>
class backward
{
  public:
    inline static vector<T, N> gradient(const equation<T, N, backward> &f, const vector<T, N> &x1, const T dx)
    {
        // Initialize backward point
        vector<T, N> x0 = x1;

        // calculate partial derivatives for each x component
        vector<T, N> out;
        for (size_t i = 0; i < N; i++)
        {
            // step backward by dx
            x0[i] -= dx;

            // evaluate derivative
            out[i] = (f(x1) - f(x0)) / dx;

            // step forward by dx
            x0[i] += dx;
        }

        // return the gradient of f
        return out;
    }
    inline static matrix<T, N, N> hessian(const equation<T, N, backward> f, const vector<T, N> &x1, const T dx)
    {
        // H_ij = d_2f/dx_i*dx_j
        matrix<T, N, N> hes;

        // Initialize backward point
        vector<T, N> x0 = x1;

        // calculate partial derivatives for each x component
        for (size_t i = 0; i < N; i++)
        {
            // step backward by dx
            x0[i] -= dx;

            // evaluate derivative
            vector<T, N> d2f_dx2 = (backward<T, N>::gradient(f, x1, dx) - backward<T, N>::gradient(f, x0, dx)) / dx;
            for (size_t j = 0; j < N; j++)
            {
                hes.get(i, j) = d2f_dx2[j];
            }

            // step forward by dx
            x0[i] += dx;
        }

        // return hessian matrix of equation
        return hes;
    }
    inline static matrix<T, N, N> jacobian(const equation<T, N, backward> f[N], const vector<T, N> &x1, const T dx)
    {
        // J_ij = df_i/dx_j
        matrix<T, N, N> jac;

        // Evaluate gradient for all functions in range [0, N)
        for (size_t i = 0; i < N; i++)
        {
            // Calculate the gradient for ith row
            const vector<T, N> grad = backward<T, N>::gradient(f[i], x1, dx);

            // Assign gradient values along this row
            for (size_t j = 0; j < N; j++)
            {
                jac.get(i, j) = grad[j];
            }
        }

        // return jacobian matrix of system
        return jac;
    }
};

// First order center finite difference
template <typename T, size_t N>
class center
{
  public:
    inline static vector<T, N> gradient(const equation<T, N, center> &f, const vector<T, N> &x1, const T dx)
    {
        // Initialize backward and forward point
        const T half_dx = dx * 0.5;
        vector<T, N> x0 = x1;
        vector<T, N> x2 = x1;

        // calculate partial derivatives for each x component
        vector<T, N> out;
        for (size_t i = 0; i < N; i++)
        {
            // step backward by half dx
            x0[i] -= half_dx;

            // step forward by half dx
            x2[i] += half_dx;

            // evaluate derivative
            out[i] = (f(x2) - f(x0)) / dx;

            // step forward by half dx
            x0[i] += half_dx;

            // step backward by half dx
            x2[i] -= half_dx;
        }

        // return the gradient of f
        return out;
    }
    inline static matrix<T, N, N> hessian(const equation<T, N, center> f, const vector<T, N> &x1, const T dx)
    {
        // H_ij = d_2f/dx_i*dx_j
        matrix<T, N, N> hes;

        // Initialize backward and forward point
        const T half_dx = dx * 0.5;
        vector<T, N> x0 = x1;
        vector<T, N> x2 = x1;

        // calculate partial derivatives for each x component
        for (size_t i = 0; i < N; i++)
        {
            // step backward by half dx
            x0[i] -= half_dx;

            // step forward by half dx
            x2[i] += half_dx;

            // evaluate derivative
            vector<T, N> d2f_dx2 = (center<T, N>::gradient(f, x1, dx) - center<T, N>::gradient(f, x0, dx)) / half_dx;
            for (size_t j = 0; j < N; j++)
            {
                hes.get(i, j) = d2f_dx2[j];
            }

            // step forward by half dx
            x0[i] += half_dx;

            // step backward by half dx
            x2[i] -= half_dx;
        }

        // return hessian matrix of equation
        return hes;
    }
    inline static matrix<T, N, N> jacobian(const equation<T, N, center> f[N], const vector<T, N> &x1, const T dx)
    {
        // J_ij = df_i/dx_j
        matrix<T, N, N> jac;

        // Evaluate gradient for all functions in range [0, N)
        for (size_t i = 0; i < N; i++)
        {
            // Calculate the gradient for ith row
            const vector<T, N> grad = center<T, N>::gradient(f[i], x1, dx);

            // Assign gradient values along this row
            for (size_t j = 0; j < N; j++)
            {
                jac.get(i, j) = grad[j];
            }
        }

        // return jacobian matrix of system
        return jac;
    }
};

// First order forward finite difference
template <typename T, size_t N>
class forward
{
  public:
    inline static vector<T, N> gradient(const equation<T, N, forward> &f, const vector<T, N> &x1, const T dx)
    {
        // Initialize forward point
        vector<T, N> x2 = x1;

        // calculate partial derivatives for each x component
        vector<T, N> out;
        for (size_t i = 0; i < N; i++)
        {
            // step forward by dx
            x2[i] += dx;

            // evaluate derivative
            out[i] = (f(x2) - f(x1)) / dx;

            // step backward by dx
            x2[i] -= dx;
        }

        // return the gradient of f
        return out;
    }
    inline static matrix<T, N, N> hessian(const equation<T, N, forward> f, const vector<T, N> &x1, const T dx)
    {
        // H_ij = d_2f/dx_i*dx_j
        matrix<T, N, N> hes;

        // Initialize forward point
        vector<T, N> x2 = x1;

        // calculate partial derivatives for each x component
        for (size_t i = 0; i < N; i++)
        {
            // step forward by dx
            x2[i] += dx;

            // evaluate derivative
            vector<T, N> d2f_dx2 = (forward<T, N>::gradient(f, x2, dx) - forward<T, N>::gradient(f, x1, dx)) / dx;
            for (size_t j = 0; j < N; j++)
            {
                hes.get(i, j) = d2f_dx2[j];
            }

            // step backward by dx
            x2[i] -= dx;
        }

        // return hessian matrix of equation
        return hes;
    }
    inline static matrix<T, N, N> jacobian(const equation<T, N, forward> f[N], const vector<T, N> &x1, const T dx)
    {
        // J_ij = df_i/dx_j
        matrix<T, N, N> jac;

        // Evaluate gradient for all functions in range [0, N)
        for (size_t i = 0; i < N; i++)
        {
            // Calculate the gradient for ith row
            const vector<T, N> grad = forward<T, N>::gradient(f[i], x1, dx);

            // Assign gradient values along this row
            for (size_t j = 0; j < N; j++)
            {
                jac.get(i, j) = grad[j];
            }
        }

        // return jacobian matrix of system
        return jac;
    }
};
}

#endif