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
#ifndef __MATRIX__
#define __MATRIX__

#include <cmath>
#include <mml/utility.h>
#include <mml/vec.h>
#include <type_traits>

namespace mml
{

// Forward declaration of matrix
template <typename T, size_t R, size_t C>
class matrix;

template <typename T, size_t R, size_t C>
class det_matrix
{
  public:
    inline static T det(const matrix<T, R, C> &mat)
    {
        matrix<T, R - 1, C - 1> sub_matrix;
        T alt = 1;
        T out = 0;
        for (size_t c = 0; c < R; c++)
        {
            size_t m = 0;
            for (size_t i = 1; i < R; i++)
            {
                size_t n = 0;
                for (size_t j = 0; j < R; j++)
                {
                    if (j != c)
                    {
                        sub_matrix.get(m, n) = mat.get(i, j);

                        // Increment the column count
                        n++;
                    }
                }

                // Increment the row count
                m++;
            }

            // Recursively calculate determinant
            out += alt * mat.get(0, c) * det_matrix<T, R - 1, C - 1>::det(sub_matrix);

            // Alternating sum
            alt *= -1.0;
        }

        // return
        return out;
    }
};

template <typename T>
class det_matrix<T, 1, 1>
{
  public:
    inline static T det(const matrix<T, 1, 1> &mat)
    {
        // determinant special case for 1x1 matrix
        return mat.get(0, 0);
    }
};

template <typename T>
class det_matrix<T, 2, 2>
{
  public:
    inline static T det(const matrix<T, 2, 2> &mat)
    {
        // determinant special case for 2x2 matrix
        return mat.get(0, 0) * mat.get(1, 1) - mat.get(1, 0) * mat.get(0, 1);
    }
};

template <typename T, size_t R, size_t C>
class matrix
{
  private:
    T _mat[R][C]; // optimized for row operations with column vector

    void inline constexpr static assert_square()
    {
        // Assert that this matrix is square
        static_assert(std::is_same<std::integral_constant<size_t, R>, std::integral_constant<size_t, C>>::value, "matrix.determinant: matrix is not square!");
    }
    inline void operator/=(const T v)
    {
        // Add all rows in matrix
        for (size_t i = 0; i < R; i++)
        {
            // For all columns in row
            for (size_t j = 0; j < C; j++)
            {
                this->get(i, j) /= v;
            }
        }
    }
    inline void decompose(size_t o[R], T s[R])
    {
        // For all rows, make s[i] the max of the ith row of the matrix
        for (size_t i = 0; i < R; i++)
        {
            o[i] = i;
            s[i] = this->get(i, 0);
            for (size_t j = 1; j < C; j++)
            {
                if (std::abs(this->get(i, j)) > s[i])
                {
                    s[i] = std::abs(this->get(i, j));
                }
            }
        }

        for (size_t k = 0; k < R - 1; k++)
        {
            // Perform matrix pivot
            this->pivot(o, s, k);

            // Check for singular matrix
            if (std::abs(this->get(o[k], k) / s[o[k]]) < 1E-4)
            {
                throw std::runtime_error("matrix.ludecomp(): singular matrix");
            }

            for (size_t i = k + 1; i < R; i++)
            {
                T factor = this->get(o[i], k) / this->get(o[k], k);
                this->get(o[i], k) = factor;
                for (size_t j = k + 1; j < C; j++)
                {
                    this->get(o[i], j) -= factor * this->get(o[k], j);
                }
            }

            // Check again for matrix singularity
            if (std::abs(this->get(o[k], k) / s[o[k]]) < 1E-4)
            {
                throw std::runtime_error("matrix.ludecomp(): singular matrix");
            }
        }
    }
    inline void pivot(size_t o[R], const T s[R], const size_t k) const
    {
        // Find the index that maximizes (o[i], k) / s[o[i]] in range [k, R]
        // and swap with o[k]
        size_t max_index = k;
        T max = std::abs(this->get(o[k], k) / s[o[k]]);
        for (size_t i = k + 1; i < R; i++)
        {
            const T value = std::abs(this->get(o[i], k) / s[o[i]]);
            if (value > max)
            {
                max = value;
                max_index = i;
            }
        }

        // Swap o[max_index] and o[k]
        mml::swap(o[max_index], o[k]);
    }
    inline vector<T, C> substitute(const size_t o[R], vector<T, C> &v) const
    {
        vector<T, C> out;

        // Forward substitution
        // Lower diagonal matrix row product
        for (size_t i = 1; i < R; i++)
        {
            T sum = v[o[i]];
            for (size_t j = 0; j < i; j++)
            {
                sum -= this->get(o[i], j) * v[o[j]];
            }
            v[o[i]] = sum;
        }

        // Back substitution
        // Initialize last element of out vector
        out[R - 1] = v[o[R - 1]] / this->get(o[R - 1], R - 1);

        // Upper diagonal matrix row product
        for (int i = R - 2; i > -1; i--)
        {
            T sum = 0.0;
            for (size_t j = i + 1; j < C; j++)
            {
                sum += this->get(o[i], j) * out[j];
            }
            out[i] = (v[o[i]] - sum) / this->get(o[i], i);
        }

        return out;
    }

  public:
    matrix()
    {
        // Add all rows in matrix
        for (size_t i = 0; i < R; i++)
        {
            // For all columns in row
            for (size_t j = 0; j < C; j++)
            {
                if (i == j)
                {
                    this->get(i, j) = 1.0;
                }
                else
                {
                    this->get(i, j) = 0.0;
                }
            }
        }
    }
    matrix(const T value)
    {
        // Add all rows in matrix
        for (size_t i = 0; i < R; i++)
        {
            // For all columns in row
            for (size_t j = 0; j < C; j++)
            {
                this->get(i, j) = value;
            }
        }
    }
    inline matrix<T, R, C> operator+(const matrix<T, R, C> &m) const
    {
        matrix<T, R, C> out;

        // Add all rows in matrix
        for (size_t i = 0; i < R; i++)
        {
            // For all columns in row
            for (size_t j = 0; j < C; j++)
            {
                out.get(i, j) = this->get(i, j) + m.get(i, j);
            }
        }

        // return the summed matrix
        return out;
    }
    inline matrix<T, R, C> operator-(const matrix<T, R, C> &m) const
    {
        matrix<T, R, C> out;

        // Add all rows in matrix
        for (size_t i = 0; i < R; i++)
        {
            // For all columns in row
            for (size_t j = 0; j < C; j++)
            {
                out.get(i, j) = this->get(i, j) - m.get(i, j);
            }
        }

        // return the summed matrix
        return out;
    }
    inline void operator+=(const matrix<T, R, C> &m)
    {
        // Add all rows in matrix
        for (size_t i = 0; i < R; i++)
        {
            // For all columns in row
            for (size_t j = 0; j < C; j++)
            {
                this->get(i, j) += m.get(i, j);
            }
        }
    }
    inline void operator-=(const matrix<T, R, C> &m)
    {
        // Add all rows in matrix
        for (size_t i = 0; i < R; i++)
        {
            // For all columns in row
            for (size_t j = 0; j < C; j++)
            {
                this->get(i, j) -= m.get(i, j);
            }
        }
    }
    inline T determinant() const
    {
        // Assert that this matrix is square R==C
        assert_square();

        // Calculate the determinant of this matrix recursively
        return det_matrix<T, R, C>::det(*this);
    }
    inline T get(const size_t i, const size_t j) const
    {
        // row/column ordering
        return _mat[i][j];
    }
    inline T &get(const size_t i, const size_t j)
    {
        // row/column ordering
        return _mat[i][j];
    }
    inline matrix<T, R, C> inverse() const
    {
        // Assert that this matrix is square R==C
        assert_square();

        matrix<T, R - 1, C - 1> sub_matrix;
        matrix<T, R, C> cofactor;

        // Check if determinant is zero
        const T determinant = this->determinant();
        if (std::abs(determinant) < 1E-4)
        {
            throw std::runtime_error("matrix.inverse(): determinant equals zero");
        }

        // Calculate the matrix of cofactors
        for (size_t p = 0; p < R; p++)
        {
            T alt = 1.0;
            for (size_t q = 0; q < C; q++)
            {
                // Make (R-1) x (R-1) minor matrix
                size_t m = 0;
                for (size_t i = 0; i < R; i++)
                {
                    // skip over this loop
                    if (i == p)
                    {
                        continue;
                    }

                    size_t n = 0;
                    for (size_t j = 0; j < C; j++)
                    {
                        // skip over this loop
                        if (j == q)
                        {
                            continue;
                        }

                        sub_matrix.get(m, n) = this->get(i, j);

                        // Increment the column count
                        n++;
                    }

                    // Increment the row count
                    m++;
                }

                // alternate sign for cofactor matrix
                alt = (p + q) % 2 == 0 ? 1 : -1;
                cofactor.get(p, q) = alt * sub_matrix.determinant();
            }
        }

        // adjugate by taking the transpose of the cofactor matrix
        matrix<T, C, R> adjugate = cofactor.transpose();

        // calculate the inverse by dividing by determinant
        adjugate /= determinant;

        // return matrix inverse
        return adjugate;
    }
    // This function solves the equation [A]{X} = {B}
    inline vector<T, C> ludecomp(const vector<T, C> &v) const
    {
        // Assert that this matrix is square R==C
        assert_square();

        // Make a local copy for manipulating decomposition
        matrix<T, R, C> A = *this;
        vector<T, C> B = v;

        // Local arrays
        size_t o[R];
        T s[R];

        // Perform decomposition
        A.decompose(o, s);

        // Calculate the solution vector
        return A.substitute(o, B);
    }
    inline matrix<T, C, R> transpose() const
    {
        matrix<T, C, R> out;

        // Copy transposed matrix into out
        for (size_t i = 0; i < R; i++)
        {
            for (size_t j = 0; j < C; j++)
            {
                out.get(j, i) = this->get(i, j);
            }
        }

        // return the transpose of this matrix
        return out;
    }
};
}

#endif