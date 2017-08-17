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
#ifndef __NEURAL_NET__
#define __NEURAL_NET__

#include <chrono>
#include <cmath>
#include <mml/vec.h>
#include <random>
#include <stdexcept>
#include <vector>

namespace mml
{

template <typename T>
class nnode
{
  private:
    T _weight;
    T _bias;
    T _output;

    static T transfer(const T input)
    {
        return 1.0 / (1.0 + std::exp(-input));
    }

  public:
    nnode() : _weight(0.0), _bias(0.0), _output(0.0) {}
    nnode(const T weight, const T bias) : _weight(weight), _bias(bias), _output(0.0) {}
    nnode<T> operator*(const nnode<T> &n) const
    {
        const T w = _weight * n._weight;
        const T b = _bias + n._bias;
        return nnode<T>(w, b);
    }
    nnode<T> &operator*=(const nnode<T> &n)
    {
        _weight *= n._weight;
        _bias += n._bias;
        return *this;
    }
    T output() const
    {
        return _output;
    }
    void sum(const T input)
    {
        _output += transfer(input * _weight + _bias);
    }
    void zero()
    {
        _output = 0.0;
    }
};

template <typename T, size_t IN, size_t OUT>
class nnet
{
  private:
    vector<T, IN> _input;
    vector<T, OUT> _output;
    std::vector<std::vector<nnode<T>>> _layers;

  public:
    nnet() {}
    void add_layer(const size_t size)
    {
        // Zero initialize layer to zero
        _layers.emplace_back(size);
    }
    static nnet<T, IN, OUT> breed(const nnet<T, IN, OUT> &p1, const nnet<T, IN, OUT> &p2)
    {
        // Initialize dimensions with p1
        nnet<T, IN, OUT> out = p1;

        // Multiply nets together
        const size_t layers = p1._layers.size();
        for (size_t i = 0; i < layers; i++)
        {
            // Multiply nets together
            const size_t nodes = p1._layers[i].size();
            for (size_t j = 0; j < nodes; j++)
            {
                out._layers[i][j] = p1._layers[i][j] * p2._layers[i][j];
            }
        }

        return out;
    }
    vector<T, OUT> calculate()
    {
        // Zero out output array
        _output.zero();

        // Zero out all layers
        zero();

        if (_layers.size() > 0)
        {
            // Map input to first layer of net
            const size_t first = _layers[0].size();
            for (size_t i = 0; i < IN; i++)
            {
                // For all nodes in first layer
                for (size_t j = 0; j < first; j++)
                {
                    _layers[0][j].sum(_input[i]);
                }
            }

            // Do N-1 propagations from first layer
            const size_t layers = _layers.size() - 1;
            for (size_t i = 0; i < layers; i++)
            {
                // For all nodes in in layer
                const size_t size_in = _layers[i].size();
                for (size_t j = 0; j < size_in; j++)
                {
                    // For all nodes in out layer
                    const size_t size_out = _layers[i + 1].size();
                    for (size_t k = 0; k < size_out; k++)
                    {
                        _layers[i + 1][k].sum(_layers[i][j].output());
                    }
                }
            }

            // Map to last layer to output of net
            const std::vector<nnode<T>> &last = _layers.back();
            const size_t output = last.size();
            for (size_t i = 0; i < OUT; i++)
            {
                // For all nodes in first layer
                for (size_t j = 0; j < output; j++)
                {
                    _output[i] += last[j].output();
                }
            }
        }
        else
        {
            _output = _input;
        }

        return _output;
    }
    static bool compatible(const nnet<T, IN, OUT> &p1, const nnet<T, IN, OUT> &p2)
    {
        // Test if nets are compatible
        if (p1._layers.size() != p2._layers.size())
        {
            throw std::runtime_error("nnet: can't breed incompatible neural nets, layers differ");
        }

        // Check net compatibility
        const size_t layers = p1._layers.size();
        for (size_t i = 0; i < layers; i++)
        {
            // For all nodes in in layer
            const size_t nodes = p1._layers[i].size();
            if (p2._layers[i].size() != nodes)
            {
                throw std::runtime_error("nnet: can't breed incompatible neural nets, nodes differ");
            }
        }

        return true;
    }
    T get_node(const size_t i, const size_t j)
    {
        return _layers[i][j].output();
    }
    void mutate()
    {
        // Breed with a randomized net
        std::uniform_real_distribution<T> dst(-100.0, 100.0);
        std::mt19937 rgen;
        const int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rgen.seed(seed);

        // For all net layers
        const size_t layers = _layers.size();
        for (size_t i = 0; i < layers; i++)
        {
            // For all layer nodes
            const size_t nodes = _layers[i].size();
            for (size_t j = 0; j < nodes; j++)
            {
                // random node
                _layers[i][j] *= nnode<T>(dst(rgen), dst(rgen));
            }
        }
    }
    void randomize()
    {
        // Random number generator
        std::uniform_real_distribution<T> dst(-1.0, 1.0);
        std::mt19937 rgen;
        const int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rgen.seed(seed);

        // For all net layers
        const size_t layers = _layers.size();
        for (size_t i = 0; i < layers; i++)
        {
            // For all layer nodes
            const size_t nodes = _layers[i].size();
            for (size_t j = 0; j < nodes; j++)
            {
                // random node
                _layers[i][j] = nnode<T>(dst(rgen), dst(rgen));
            }
        }
    }
    void set_input(const vector<T, IN> &input)
    {
        _input = input;
    }
    void zero()
    {
        // For all net layers
        const size_t layers = _layers.size();
        for (size_t i = 0; i < layers; i++)
        {
            // For all layer nodes
            const size_t nodes = _layers[i].size();
            for (size_t j = 0; j < nodes; j++)
            {
                // zero output
                _layers[i][j].zero();
            }
        }
    }
};
}

#endif
