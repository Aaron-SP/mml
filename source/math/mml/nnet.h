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
#include <functional>
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
        const T product = std::abs(_weight * n._weight);

        // Negative weights are dominant
        T sign = 1.0;
        if (_weight < 0.0 || n._weight < 0.0)
        {
            sign = -1.0;
        }

        // geometric mean and average
        const T w = sign * std::sqrt(product);
        const T b = (_bias + n._bias) * 0.5;
        return nnode<T>(w, b);
    }
    nnode<T> &operator*=(const nnode<T> &n)
    {
        const T product = std::abs(_weight * n._weight);

        // Negative weights are dominant
        T sign = 1.0;
        if (_weight < 0.0 || n._weight < 0.0)
        {
            sign = -1.0;
        }

        // geometric mean and average
        _weight = sign * std::sqrt(product);
        _bias = (_bias + n._bias) * 0.5;
        return *this;
    }
    T get_bias() const
    {
        return _bias;
    }
    T get_weight() const
    {
        return _weight;
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
    bool _final;

    void on_net(const std::function<void(nnode<T> &, const size_t, const size_t)> &f)
    {
        // For all net layers
        const size_t layers = _layers.size();
        for (size_t i = 0; i < layers; i++)
        {
            // For all layer nodes
            const size_t nodes = _layers[i].size();
            for (size_t j = 0; j < nodes; j++)
            {
                f(_layers[i][j], i, j);
            }
        }
    }
    void on_const_net(const std::function<void(const nnode<T> &, const size_t, const size_t)> &f) const
    {
        // For all net layers
        const size_t layers = _layers.size();
        for (size_t i = 0; i < layers; i++)
        {
            // For all layer nodes
            const size_t nodes = _layers[i].size();
            for (size_t j = 0; j < nodes; j++)
            {
                f(_layers[i][j], i, j);
            }
        }
    }
    void zero_output()
    {
        // Zero out all layers
        const auto f = [](nnode<T> &node, const size_t i, const size_t j) {
            node.zero();
        };

        // Randomize the net
        on_net(f);
    }

  public:
    nnet() : _final(false) {}
    void add_layer(const size_t size)
    {
        if (!_final)
        {
            // Zero initialize layer to zero
            _layers.emplace_back(size);
        }
        else
        {
            throw std::runtime_error("nnet: can't add layers to a finalized neural net");
        }
    }
    static nnet<T, IN, OUT> breed(const nnet<T, IN, OUT> &p1, const nnet<T, IN, OUT> &p2)
    {
        // Initialize dimensions with p1
        nnet<T, IN, OUT> out = p1;

        const auto f = [&p1, &p2](nnode<T> &node, const size_t i, const size_t j) {
            node = p1._layers[i][j] * p2._layers[i][j];
        };

        // Breed nets together
        out.on_net(f);

        return out;
    }
    vector<T, OUT> calculate()
    {
        // finalize the net, can't add layers after calling this function
        finalize();

        // Zero out all node output
        zero_output();

        // If we added any layers
        if (_layers.size() > 1)
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

            // Map last layer to output of net
            // Last layer is special and added during finalize so we can just grab the output value
            // From the last internal layer for the output
            const std::vector<nnode<T>> &last = _layers.back();
            for (size_t i = 0; i < OUT; i++)
            {
                _output[i] = last[i].output();
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
    const vector<T, IN> &get_input() const
    {
        return _input;
    }
    T get_node(const size_t i, const size_t j)
    {
        return _layers[i][j].output();
    }
    void finalize()
    {
        if (!_final)
        {
            // Create output network layer
            _layers.emplace_back(OUT);
            _final = true;
        }
    }
    void mutate()
    {
        std::uniform_real_distribution<T> dst(-10.0, 10.0);
        std::mt19937 rgen;
        const int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rgen.seed(seed);

        const auto f = [&dst, &rgen](nnode<T> &node, const size_t i, const size_t j) {
            node *= nnode<T>(dst(rgen), dst(rgen));
        };

        // Breed with a randomized net
        on_net(f);
    }
    void randomize()
    {
        // Random number generator
        std::uniform_real_distribution<T> dst(-1.0, 1.0);
        std::mt19937 rgen;
        const int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rgen.seed(seed);

        const auto f = [&dst, &rgen](nnode<T> &node, const size_t i, const size_t j) {
            node = nnode<T>(dst(rgen), dst(rgen));
        };

        // Randomize the net
        on_net(f);
    }
    void set_input(const vector<T, IN> &input)
    {
        _input = input;
    }
    std::vector<T> serialize() const
    {
        std::vector<T> out;

        // Serial net dimensions
        out.push_back((T)IN);
        out.push_back((T)OUT);
        out.push_back((T)_layers.size());

        // Serialize layer sizes
        const size_t size = _layers.size();
        for (size_t i = 0; i < size; i++)
        {
            out.push_back((T)_layers[i].size());
        }

        // Serialize net data
        const auto f = [&out](const nnode<T> &node, const size_t i, const size_t j) {
            out.push_back(node.get_weight());
            out.push_back(node.get_bias());
        };

        // Randomize the net
        on_const_net(f);

        return out;
    }
    void deserialize(const std::vector<T> &data)
    {
        // Use int here, in case someone feeds in garbage data
        // Check input size
        const int in = (int)data[0];
        if (in != IN)
        {
            throw std::runtime_error("nnet: can't deserialize, expected input '" + std::to_string(IN) + "' but got '" + std::to_string(in) + "'");
        }

        // Check output size
        const int out = (int)data[1];
        if (out != OUT)
        {
            throw std::runtime_error("nnet: can't deserialize, expected input '" + std::to_string(OUT) + "' but got '" + std::to_string(out) + "'");
        }

        // Clear the layers
        _layers.clear();
        int count = 0;
        const int size = (int)data[2];
        for (int i = 0; i < size; i++)
        {
            // Add layers of length
            const int length = (int)data[3 + i];
            this->add_layer(length);

            // Count nodes
            count += length;
        }

        // Check that the number of nodes makes sense
        const size_t left = data.size() - 3 - size;
        if (count * 2.0 != left)
        {
            throw std::runtime_error("nnet: can't deserialize node mismatch");
        }

        // Starting index, assign values to net
        size_t index = 3 + size;
        const auto f = [&data, &index](nnode<T> &node, const size_t i, const size_t j) {
            node = nnode<T>(data[index], data[index + 1]);
            index += 2;
        };

        // Randomize the net
        on_net(f);

        // Finalize this network
        _final = true;
    }
};
}

#endif
