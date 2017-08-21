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
    std::vector<T> _weights;
    std::vector<T> _inputs;
    T _bias;
    T _sum;
    T _output;
    T _delta;

    inline static T transfer(const T input)
    {
        return 1.0 / (1.0 + std::exp(-input));
    }

    inline void zero()
    {
        _sum = _bias;
        _output = 0.0;
        _delta = 0.0;
    }

  public:
    nnode(const size_t size) : _weights(size, 0.0), _inputs(size, 0.0), _bias(0.0), _sum(0.0), _output(0.0), _delta(0.0) {}
    nnode(const std::vector<T> &weights, const T bias) : _weights(weights), _inputs(weights.size(), 0.0), _bias(bias), _sum(0.0), _output(0.0), _delta(0.0) {}
    inline nnode<T> operator*(const nnode<T> &n) const
    {
        const size_t size = n._weights.size();
        std::vector<T> weights(size, 0);
        for (size_t i = 0; i < size; i++)
        {
            const T product = std::abs(_weights[i] * n._weights[i]);

            // Negative weights are dominant
            T sign = 1.0;
            if (_weights[i] < 0.0 || n._weights[i] < 0.0)
            {
                sign = -1.0;
            }

            // geometric mean
            weights[i] = sign * std::sqrt(product);
        }

        // average
        const T b = (_bias + n._bias) * 0.5;
        return nnode<T>(weights, b);
    }
    inline nnode<T> &operator*=(const nnode<T> &n)
    {
        const size_t size = n._weights.size();
        for (size_t i = 0; i < size; i++)
        {
            const T product = std::abs(_weights[i] * n._weights[i]);

            // Negative weights are dominant
            T sign = 1.0;
            if (_weights[i] < 0.0 || n._weights[i] < 0.0)
            {
                sign = -1.0;
            }

            // geometric mean
            _weights[i] = sign * std::sqrt(product);
        }

        // average
        _bias = (_bias + n._bias) * 0.5;
        return *this;
    }
    inline void calculate()
    {
        // Reset the node
        zero();

        // Calculate transfer
        _output = transfer(_sum);
    }
    T delta(const size_t index) const
    {
        // dk * Wjk in backprop
        return _delta * _weights[index];
    }
    void backprop(const T propagated)
    {
        // propagated = sum(dk * Wjk) or if last layer (Ok - tk)
        // dj = (Oj)*(1.0 - Oj) * propagated
        _delta = _output * (1.0 - _output) * propagated;

        // Calculate step, step size 0.1
        const T step = -0.1 * _delta;

        // Update weights
        const size_t size = _inputs.size();
        for (size_t i = 0; i < size; i++)
        {
            // Update weights
            _weights[i] += step * _inputs[i];
        }

        // Update bias
        _bias += step;
    }
    inline T get_bias() const
    {
        return _bias;
    }
    inline size_t get_inputs() const
    {
        return _weights.size();
    }
    inline std::vector<T> get_weights() const
    {
        return _weights;
    }
    inline T output() const
    {
        return _output;
    }
    inline void sum(const T input, const size_t index)
    {
        // Store input for later
        _inputs[index] = input;

        // Sum input
        _sum += input * _weights[index];
    }
};

template <typename T>
class net_rng
{
  private:
    std::uniform_real_distribution<T> _mut_dist;
    std::uniform_real_distribution<T> _ran_dist;
    std::uniform_int_distribution<int> _int_dist;
    std::mt19937 _rgen;

  public:
    net_rng()
        : _mut_dist(-10.0, 10.0),
          _ran_dist(-1.0, 1.0),
          _int_dist(0, 100),
          _rgen(std::chrono::high_resolution_clock::now().time_since_epoch().count())
    {
    }
    net_rng(const std::uniform_real_distribution<T> &mut_dist,
            const std::uniform_real_distribution<T> &ran_dist,
            const std::uniform_int_distribution<int> &int_dist)
        : _mut_dist(mut_dist),
          _ran_dist(ran_dist),
          _int_dist(int_dist),
          _rgen(std::chrono::high_resolution_clock::now().time_since_epoch().count())
    {
    }
    T mutation()
    {
        return _mut_dist(_rgen);
    }
    std::vector<T> mutation(size_t size)
    {
        // Create a vector of random numbers
        std::vector<T> out(size);
        for (size_t i = 0; i < size; i++)
        {
            out[i] = this->mutation();
        }

        return out;
    }
    T random()
    {
        return _ran_dist(_rgen);
    }
    std::vector<T> random(size_t size)
    {
        // Create a vector of random numbers
        std::vector<T> out(size);
        for (size_t i = 0; i < size; i++)
        {
            out[i] = this->random();
        }

        return out;
    }
    int random_int()
    {
        return _int_dist(_rgen);
    }
    void reseed()
    {
        _rgen.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
};

template <typename T>
class nnlayer
{
  private:
    std::vector<nnode<T>> _nodes;
    size_t _inputs;

  public:
    nnlayer(const size_t size, const size_t inputs) : _nodes(size, inputs), _inputs(inputs) {}
    inline size_t size() const
    {
        return _nodes.size();
    }
    inline size_t inputs() const
    {
        return _inputs;
    }
    inline nnode<T> &operator[](const size_t n)
    {
        return _nodes[n];
    }
    inline const nnode<T> &operator[](const size_t n) const
    {
        return _nodes[n];
    }
};

template <typename T, size_t IN, size_t OUT>
class nnet
{
  private:
    vector<T, IN> _input;
    vector<T, OUT> _output;
    std::vector<nnlayer<T>> _layers;
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

  public:
    nnet() : _final(false)
    {
        // Initialize first layer
        add_layer(IN);
    }
    void add_layer(const size_t size)
    {
        if (!_final)
        {
            // If first layer
            size_t inputs = 1;
            if (_layers.size() != 0)
            {
                // Size of last layer is number of inputs to next layer
                inputs = _layers.back().size();
            }

            // Zero initialize layer to zero
            _layers.emplace_back(size, inputs);
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
    void backprop(const vector<T, OUT> &set_point)
    {
        // We assume the network is in calculated state

        // If we are in a valid state
        if (_layers.size() >= 2)
        {
            // Do backprop for last layer first
            const size_t last = _layers.size() - 1;

            // Check that last layer is the appropriate size
            const size_t last_size = _layers[last].size();
            if (last_size != OUT)
            {
                throw std::runtime_error("nnet: backprop invalid output dimension");
            }

            // For all nodes in last layer
            for (size_t i = 0; i < last_size; i++)
            {
                const T error = _layers[last][i].output() - set_point[i];

                // Do backprop for node in last layer
                _layers[last][i].backprop(error);
            }

            // For all internal layers, iterating backwards
            const size_t layers = _layers.size();
            for (size_t i = 1; i < layers; i++)
            {
                size_t current = last - i;
                const size_t nodes = _layers[current].size();
                for (size_t j = 0; j < nodes; j++)
                {
                    // For all nodes in layer, calculate delta summation
                    T sum = 0.0;
                    const size_t size_out = _layers[current + 1].size();
                    for (size_t k = 0; k < size_out; k++)
                    {
                        sum += _layers[current + 1][k].delta(j);
                    }

                    // Do backprop for this node
                    _layers[current][j].backprop(sum);
                }
            }
        }
    }
    vector<T, OUT> calculate()
    {
        // finalize the net, can't add layers after calling this function
        finalize();

        // If we added any layers
        if (_layers.size() > 2)
        {
            // Map input to first layer of net
            for (size_t i = 0; i < IN; i++)
            {
                _layers[0][i].sum(_input[i], 0);
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
                        _layers[i][j].calculate();
                        _layers[i + 1][k].sum(_layers[i][j].output(), j);
                    }
                }
            }

            // Map last layer to output of net
            // Last layer is special and added during finalize so we can just grab the output value
            // From the last internal layer for the output
            nnlayer<T> &last = _layers.back();
            for (size_t i = 0; i < OUT; i++)
            {
                last[i].calculate();
                _output[i] = last[i].output();
            }
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
            // Create output network layer with input count from last layer
            add_layer(OUT);
            _final = true;
        }
    }
    void mutate(mml::net_rng<T> &ran)
    {
        const auto f = [&ran](nnode<T> &node, const size_t i, const size_t j) {
            const size_t inputs = node.get_inputs();
            node *= nnode<T>(ran.mutation(inputs), ran.mutation());
        };

        // Breed with a randomized net
        on_net(f);
    }
    void randomize(mml::net_rng<T> &ran)
    {
        const auto f = [&ran](nnode<T> &node, const size_t i, const size_t j) {
            const size_t inputs = node.get_inputs();
            node = nnode<T>(ran.random(inputs), ran.random());
        };

        // Randomize the net
        on_net(f);
    }
    void reset()
    {
        // Clear layers
        _layers.clear();

        // Unfinalize the net
        _final = false;

        // Initialize first layer
        add_layer(IN);
    }
    void set_input(const vector<T, IN> &input)
    {
        _input = input;
    }
    std::vector<T> serialize() const
    {
        std::vector<T> out;

        // Serial net dimensions
        out.push_back(static_cast<T>(IN));
        out.push_back(static_cast<T>(OUT));
        out.push_back(static_cast<T>(_layers.size()));

        // Serialize layer sizes
        const size_t size = _layers.size();
        for (size_t i = 0; i < size; i++)
        {
            const size_t nodes = _layers[i].size();
            out.push_back(static_cast<T>(nodes));
        }

        // Serialize net data
        const auto f = [&out](const nnode<T> &node, const size_t i, const size_t j) {
            // Serialize all weights
            const size_t size = node.get_inputs();
            const std::vector<T> &weights = node.get_weights();
            for (size_t k = 0; k < size; k++)
            {
                out.push_back(weights[k]);
            }

            // Serialize bias
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
        const int in = static_cast<int>(data[0]);
        if (in != IN)
        {
            throw std::runtime_error("nnet: can't deserialize, expected input '" + std::to_string(IN) + "' but got '" + std::to_string(in) + "'");
        }

        // Check output size
        const int out = static_cast<int>(data[1]);
        if (out != OUT)
        {
            throw std::runtime_error("nnet: can't deserialize, expected input '" + std::to_string(OUT) + "' but got '" + std::to_string(out) + "'");
        }

        // Clear the layers
        _layers.clear();
        const int size = static_cast<int>(data[2]);

        // Check first layer size special case
        const int first = data[3];
        if (first != IN)
        {
            throw std::runtime_error("nnet: can't deserialize, expected input '" + std::to_string(IN) + "' but got '" + std::to_string(first) + "'");
        }

        // Check last layer size special case
        const int last = data[2 + size];
        if (last != OUT)
        {
            throw std::runtime_error("nnet: can't deserialize, expected input '" + std::to_string(OUT) + "' but got '" + std::to_string(last) + "'");
        }

        // Count bytes
        size_t count = 0;
        int inputs = 0;
        for (int i = 0; i < size; i++)
        {
            // Number of nodes in layer
            const int length = static_cast<int>(data[3 + i]);
            if (length <= 0)
            {
                throw std::runtime_error("nnet: invalid layer size");
            }

            // Add new layer
            this->add_layer(length);

            // Count data members in nodes
            if (i == 0)
            {
                // Bias and one input
                count += IN * 2;
                inputs = IN;
            }
            else
            {
                // Bias and inputs per node in layer
                count += length * (inputs + 1);
                inputs = length;
            }
        }

        // Check that the number of nodes makes sense
        const size_t left = data.size() - (size + 3);
        if (count != left)
        {
            throw std::runtime_error("nnet: can't deserialize node mismatch");
        }

        // Starting index, assign values to net
        size_t index = 3 + size;
        const auto f = [&data, &index](nnode<T> &node, const size_t i, const size_t j) {
            // Copy all weights from data to nnode
            const size_t size = node.get_inputs();
            std::vector<T> weights(size);
            for (size_t k = 0; k < size; k++)
            {
                weights[k] = data[index];
                index++;
            }
            node = nnode<T>(weights, data[index]);
            index++;
        };

        // Randomize the net
        on_net(f);

        // Finalize this network
        _final = true;
    }
};
}

#endif
