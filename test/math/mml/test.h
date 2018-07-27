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
#ifndef __TESTUTIL__
#define __TESTUTIL__

#include <algorithm>
#include <cmath>
#include <stdexcept>

template <typename T>
struct identity
{
    typedef T type;
};
template <typename T>
using identity_t = typename identity<T>::type;

template <typename T>
bool compare(const T one, const identity_t<T> two)
{
    return one == two;
}
template <typename T>
bool compare(const T one, const identity_t<T> two, const identity_t<T> threshold)
{
    return std::abs(one - two) <= threshold;
}
template <typename T>
bool test(const T one, const identity_t<T> two, const char *fail)
{
    const bool out = compare<T>(one, two);
    if (!out)
    {
        throw std::runtime_error(fail);
    }
    return out;
}
template <typename T>
bool test(const T one, const identity_t<T> two, const identity_t<T> tol, const char *fail)
{
    const bool out = compare(one, two, tol);
    if (!out)
    {
        throw std::runtime_error(fail);
    }
    return out;
}
template <typename T>
bool not_test(const T one, const identity_t<T> two, const char *fail)
{
    const bool out = compare(one, two);
    if (out)
    {
        throw std::runtime_error(fail);
    }
    return !out;
}
template <typename T>
bool not_test(const T one, const identity_t<T> two, const identity_t<T> tol, const char *fail)
{
    const bool out = compare(one, two, tol);
    if (out)
    {
        throw std::runtime_error(fail);
    }
    return !out;
}
#endif
