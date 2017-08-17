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
#ifndef __VECTOR__
#define __VECTOR__

#include <cmath>

namespace mml
{

template <typename T, size_t N>
class vector
{
  private:
    T _vec[N]; // column vector

  public:
    vector()
    {
        // zero all fields
        zero();
    }
    vector(const T value[N])
    {
        for (size_t i = 0; i < N; i++)
        {
            _vec[i] = value[i];
        }
    }
    vector(const T value)
    {
        for (size_t i = 0; i < N; i++)
        {
            _vec[i] = value;
        }
    }
    inline T operator[](const size_t n) const
    {
        // value getter
        return _vec[n];
    }
    inline T &operator[](const size_t n)
    {
        // value setter
        return _vec[n];
    }
    inline vector<T, N> operator+(const vector<T, N> &vec) const
    {
        vector<T, N> out;

        for (size_t i = 0; i < N; i++)
        {
            out[i] = this->operator[](i) + vec[i];
        }

        return out;
    }
    inline vector<T, N> operator-(const vector<T, N> &vec) const
    {
        vector<T, N> out;

        for (size_t i = 0; i < N; i++)
        {
            out[i] = this->operator[](i) - vec[i];
        }

        return out;
    }
    inline vector<T, N> operator*(const vector<T, N> &vec) const
    {
        vector<T, N> out;

        for (size_t i = 0; i < N; i++)
        {
            out[i] = this->operator[](i) * vec[i];
        }

        return out;
    }
    inline vector<T, N> operator/(const vector<T, N> &vec) const
    {
        vector<T, N> out;

        for (size_t i = 0; i < N; i++)
        {
            out[i] = this->operator[](i) / vec[i];
        }

        return out;
    }
    inline void operator+=(const vector<T, N> &vec)
    {
        for (size_t i = 0; i < N; i++)
        {
            this->operator[](i) += vec[i];
        }
    }
    inline void operator-=(const vector<T, N> &vec)
    {
        for (size_t i = 0; i < N; i++)
        {
            this->operator[](i) -= vec[i];
        }
    }
    inline void operator*=(const vector<T, N> &vec)
    {
        for (size_t i = 0; i < N; i++)
        {
            this->operator[](i) *= vec[i];
        }
    }
    inline void operator/=(const vector<T, N> &vec)
    {
        for (size_t i = 0; i < N; i++)
        {
            this->operator[](i) /= vec[i];
        }
    }
    inline T square_magnitude() const
    {
        T out = 0.0;

        // Calculate the square magnitude of the vector
        for (size_t i = 0; i < N; i++)
        {
            out += _vec[i] * _vec[i];
        }

        // return the square magnitude
        return out;
    }
    inline void zero()
    {
        for (size_t i = 0; i < N; i++)
        {
            _vec[i] = 0.0;
        }
    }
};
}

#endif
