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
#ifndef __NEURAL_NET_AUGMENTED__
#define __NEURAL_NET_AUGMENTED__

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <mml/nn.h>
#include <mml/vec.h>
#include <vector>

namespace mml
{

template <typename T>
class nanode
{
  private:
    std::vector<size_t> _edges;
    std::map<size_t, T> _weights;
    T _bias;
    mutable T _sum;
    mutable T _output;

    inline static T transfer_sigmoid(const T input)
    {
        const T arg = -1.0 * input;
        return 1.0 / (1.0 + std::exp(arg));
    }

  public:
    nanode()
        : _bias(0.0), _sum(0.0), _output(0.0) {}
    nanode(const T output)
        : _bias(0.0), _sum(0.0), _output(output) {}
    nanode(const std::vector<T> &data, size_t &start) : _bias(0.0), _sum(0.0), _output(0.0)
    {
        // Test if we have enough data
        if (data.size() < 3)
        {
            throw std::runtime_error("nanode: can't deserialize, not enough data");
        }

        // Get stream sizes
        const int edge_size = static_cast<int>(data[start]);
        const int weight_size = static_cast<int>(data[start + 1]);

        // Test valid sizes
        if (edge_size < 0 || weight_size < 0)
        {
            throw std::runtime_error("nanode: can't deserialize, invalid stream size");
        }

        // Record data properties
        _bias = data[start + 2];

        // Test if we have enough data
        const size_t needed = edge_size + (weight_size * 2);
        if ((data.size() - start - 3) < needed)
        {
            throw std::runtime_error("nanode: can't deserialize, invalid data size");
        }

        // Read edge data
        const size_t edge_offset = start + 3;
        _edges.reserve(edge_size);
        for (int i = 0; i < edge_size; i++)
        {
            // read edge data
            const int index = static_cast<int>(data[edge_offset + i]);
            if (index < 0)
            {
                throw std::runtime_error("nanode: can't deserialize, invalid connection index");
            }

            // push_back connection
            _edges.push_back(index);
        }

        // Read weight data
        const size_t w_offset = edge_offset + edge_size;
        for (int i = 0; i < weight_size; i++)
        {
            // Calculate read offset
            const size_t index = data[w_offset + (i * 2)];
            const T weight = data[w_offset + (i * 2) + 1];

            // Insert weight into map
            _weights.insert(std::make_pair(index, weight));
        }

        // Increment data pointer
        start += 3 + edge_size + weight_size * 2;
    }
    void calculate() const
    {
        _output = transfer_sigmoid(_sum);
    }
    void connect_edge(const size_t index)
    {
        _edges.push_back(index);
    }
    bool connect_weight(const T weight, const size_t index)
    {
        // Insert connection into node
        const auto p = _weights.insert(std::make_pair(index, weight));

        // return if success
        return p.second;
    }
    void fixed(const T out) const
    {
        _output = out;
    }
    void mutate(mml::net_rng<T> &ran)
    {
        // Don't mutate if no connections
        if (_weights.size() == 0)
        {
            return;
        }

        // Calculate a mutation type
        const unsigned r = ran.random_int();

        // Calculate map random index iterator
        const unsigned index = ran.random_int() % _weights.size();

        // Advanced iterator
        auto iter = _weights.begin();
        std::advance(iter, index);

        // Mutate the node based on type
        if (r % 2 == 0)
        {
            iter->second *= ran.mutation();
        }
        else if (r % 3 == 0)
        {
            // Mutate the bias with mult
            _bias *= ran.mutation();
        }
        else if (r % 5 == 0)
        {
            iter->second += ran.mutation();
        }
        else if (r % 7 == 0)
        {
            // Mutate the bias with add
            _bias += ran.mutation();
        }
        else if (r % 11 == 0)
        {
            iter->second -= ran.mutation();
        }
        else if (r % 13 == 0)
        {
            // Mutate the bias with sub
            _bias -= ran.mutation();
        }
    }
    inline nanode<T> &operator*=(const nanode<T> &n)
    {
        // Check the weights between species
        const size_t size = std::min(_weights.size(), n._weights.size());

        // Get weight iterators
        auto it1 = _weights.begin();
        auto it2 = n._weights.begin();

        // Perform weight and bias crossover
        for (size_t i = 0; i < size; i++)
        {
            const float &w1 = it1->second;
            const float &w2 = it2->second;
            const T product = std::abs(w1 * w2);

            // Negative weights are dominant
            T sign = 1.0;
            if (w1 < 0.0 || w2 < 0.0)
            {
                sign = -1.0;
            }

            // geometric mean weights
            it1->second = sign * std::sqrt(product);

            // Increment iterators
            it1++;
            it2++;
        }

        // average bias
        _bias = (_bias + n._bias) * 0.5;

        // return this node
        return *this;
    }
    void remove_edge(const size_t index)
    {
        // connections can occur multiple times
        _edges.erase(std::remove(_edges.begin(), _edges.end(), index), _edges.end());
    }
    void remove_weight(const size_t index)
    {
        // Check if key exists and erase
        const auto w_iter = _weights.find(index);
        if (w_iter != _weights.end())
        {
            // Only one weight per connection since map is unique
            _weights.erase(w_iter);
        }
    }
    const std::vector<size_t> &get_edges() const
    {
        return _edges;
    }
    T output() const
    {
        return _output;
    }
    void randomize(mml::net_rng<T> &ran)
    {
        // Randomize all weights
        for (auto &p : _weights)
        {
            p.second = ran.random();
        }

        // Randomize bias
        _bias = ran.random();
    }
    void reset() const
    {
        _sum = _bias;
    }
    void serialize(std::vector<T> &data) const
    {
        // Get stream sizes
        const size_t edge_size = _edges.size();
        const size_t weight_size = _weights.size();

        // Serialize data
        data.push_back(static_cast<T>(edge_size));
        data.push_back(static_cast<T>(weight_size));
        data.push_back(_bias);

        // Serialize edges
        for (size_t i = 0; i < edge_size; i++)
        {
            data.push_back(static_cast<T>(_edges[i]));
        }

        // Serialize weights
        for (const auto &p : _weights)
        {
            data.push_back(static_cast<T>(p.first));
            data.push_back(p.second);
        }
    }
    void sum(const T input, const size_t index) const
    {
        // Check errors here
        if (_weights.count(index) == 0)
        {
            throw std::runtime_error("nanode: node is disjoint");
        }

        // Get weight
        const auto w_iter = _weights.find(index);

        // Sum the input node
        _sum += input * (w_iter->second);
    }
};

template <typename T, size_t IN, size_t OUT>
class nneat
{
  private:
    std::vector<nanode<T>> _nodes;
    mutable vector<T, OUT> _output;
    unsigned _q;
    unsigned _r;
    unsigned _s;

    bool output_node(const size_t to)
    {
        bool out = false;

        if (to < (OUT + IN) && to >= IN)
        {
            return true;
        }

        return out;
    }
    bool prevent_cycles(const size_t from, const size_t to) const
    {
        bool out = true;

        // Can't connect output to anything
        if (from < (OUT + IN) && from >= IN)
        {
            return false;
        }
        // Can't connect anything to input
        else if (to < IN && to >= 0)
        {
            return false;
        }
        // Can connect to output
        else if (to < (OUT + IN) && to >= IN)
        {
            return true;
        }
        else if (from >= to)
        {
            return false;
        }

        return out;
    }

  public:
    nneat() : _q(29.0), _r(7.0), _s(3.0)
    {
        // Create input nodes
        for (size_t i = 0; i < IN; i++)
        {
            // Create an input node with constant output
            _nodes.emplace_back(0.0);
        }

        // Create output nodes
        for (size_t i = 0; i < OUT; i++)
        {
            // Create an input node with constant output
            _nodes.emplace_back();
        }
    }
    inline void add_connection(const size_t from, const size_t to)
    {
        // Check connections are valid
        if (!prevent_cycles(from, to))
        {
            return;
        }

        // Try to connect weight between from and to, default weight 1.0
        const bool inserted = _nodes[to].connect_weight(1.0, from);
        if (!inserted)
        {
            // duplicate key, so don't add edge
            return;
        }

        // Add edge between
        _nodes[from].connect_edge(to);
    }
    inline void add_node_between(const size_t from, const size_t to)
    {
        // 'to' must be an output node
        if (!output_node(to))
        {
            return;
        }

        // Check connections are valid
        if (!prevent_cycles(from, to))
        {
            return;
        }

        // Add unconnected node
        _nodes.emplace_back();

        // Get new node index
        const size_t last = _nodes.size() - 1;

        // Remove edge to 'to'
        _nodes[from].remove_edge(to);

        // Remove weight on 'to'
        _nodes[to].remove_weight(from);

        // Connect from to new node
        add_connection(from, last);

        // Connect new node to 'to'
        add_connection(last, to);
    }
    inline static nneat<T, IN, OUT> breed(const nneat<T, IN, OUT> &p1, const nneat<T, IN, OUT> &p2)
    {
        // Initialize dimensions with p1
        nneat<T, IN, OUT> out = p1;

        // Recursive traversal of DAG
        std::function<void(const nanode<T> &node1, const nanode<T> &node2)> dfs;
        dfs = [&dfs, &out, &p2](const nanode<T> &node1, const nanode<T> &node2) {

            // Get all edges on nodes
            const std::vector<size_t> &e1 = node1.get_edges();
            const std::vector<size_t> &e2 = node2.get_edges();
            const size_t edges = std::min(e1.size(), e2.size());
            for (size_t i = 0; i < edges; i++)
            {
                // Get edge index
                const size_t i1 = e1[i];
                const size_t i2 = e2[i];

                // If these are the same node in DAG
                if (i1 == i2)
                {
                    // Get next nodes
                    nanode<T> &en1 = out._nodes[i1];
                    const nanode<T> &en2 = p2._nodes[i2];

                    // Perform cross over
                    en1 *= en2;

                    // Recurse
                    dfs(en1, en2);
                }
            }
        };

        // Crossover node by node
        for (size_t i = 0; i < IN; i++)
        {
            // Get next nodes
            const nanode<T> &node1 = p1._nodes[i];
            const nanode<T> &node2 = p2._nodes[i];

            // Call recursion
            dfs(node1, node2);
        }

        return out;
    }
    inline vector<T, OUT> calculate() const
    {
        const size_t nodes = _nodes.size();
        for (size_t i = 0; i < nodes; i++)
        {
            // reset all nodes
            _nodes[i].reset();
        }

        // DO NOT CALCULATE INPUTS THEY ARE FIXED OUTPUT NODES
        for (size_t i = 0; i < IN; i++)
        {
            // Get next node
            const nanode<T> &node = _nodes[i];

            // Get node destinations
            const std::vector<size_t> &edges = node.get_edges();
            for (const auto e : edges)
            {
                _nodes[e].sum(node.output(), i);
            }
        }

        // For all nodes propagate values to output
        // Output will be skipped over since it has no edges
        const size_t node_beg = IN + OUT;
        for (size_t i = node_beg; i < nodes; i++)
        {
            // Get next node
            const nanode<T> &node = _nodes[i];

            // Calculate node
            _nodes[i].calculate();

            // Get node destinations
            const std::vector<size_t> &edges = node.get_edges();
            for (const auto e : edges)
            {
                _nodes[e].sum(node.output(), i);
            }
        }

        // Calculate output nodes
        for (size_t i = 0; i < OUT; i++)
        {
            _nodes[IN + i].calculate();
            _output[i] = _nodes[IN + i].output();
        }

        return _output;
    }
    void debug_connections() const
    {
        // print out connection order
        const size_t nodes = _nodes.size();
        for (size_t i = 0; i < nodes; i++)
        {
            // Get next node
            const nanode<T> &node = _nodes[i];

            if (i < IN)
            {
                std::cout << "input " << i << std::endl;
            }
            else if (i < (OUT + IN))
            {
                std::cout << "output " << i << std::endl;
            }
            else
            {
                std::cout << "node " << i << std::endl;
            }

            // Get node destinations
            const std::vector<size_t> &edges = node.get_edges();
            for (const auto e : edges)
            {
                std::cout << "    -> " << e << std::endl;
            }

            // Write node output
            std::cout << "    value: " << node.output() << std::endl;
        }
    }
    size_t id() const
    {
        return _nodes.size();
    }
    inline void mutate_topology(mml::net_rng<T> &ran)
    {
        // Roll dice
        const unsigned r = ran.random_int();

        // Between IN and END, even though OUT can't be a 'from'
        const unsigned from = ran.random_int() % _nodes.size();

        // Choose mutation type, connections are more probable
        if (r % _r == 0)
        {
            // 'to' must be an output, enforce only OUT
            const unsigned out = (ran.random_int() % OUT) + IN;
            add_node_between(from, out);
        }
        else
        {
            // 'to' can't be an input
            const unsigned not_input_size = _nodes.size() - IN;

            // Between OUT and END
            const unsigned to = (ran.random_int() % not_input_size) + IN;
            add_connection(from, to);
        }
    }
    inline void mutate_weight(mml::net_rng<T> &ran)
    {
        // Calculate a random layer index
        const unsigned not_input_size = _nodes.size() - IN;

        // Protect input nodes that have NO WEIGHTS
        // Between OUT and END
        const unsigned node_index = (ran.random_int() % not_input_size) + IN;

        // Mutate this node
        _nodes[node_index].mutate(ran);
    }
    inline void mutate(mml::net_rng<T> &ran)
    {
        const unsigned q = ran.random_int();
        const unsigned r = ran.random_int();
        const unsigned s = ran.random_int();

        // choose mutation type, weight mutations are more probable
        if ((q % _q == 0) && (r % _r == 0) && (s % _s == 0))
        {
            mutate_topology(ran);
        }
        else
        {
            mutate_weight(ran);
        }
    }
    inline void randomize(mml::net_rng<T> &ran)
    {
        // Skip input nodes since no weights
        const size_t node_beg = IN;
        const size_t nodes = _nodes.size();

        // randomize all hidden nodes
        for (size_t i = node_beg; i < nodes; i++)
        {
            _nodes[i].randomize(ran);
        }
    }
    inline void set_input(const vector<T, IN> &input) const
    {
        // Set input nodes
        for (size_t i = 0; i < IN; i++)
        {
            // Set fixed output for input layer
            _nodes[i].fixed(input[i]);
        }
    }
    void set_topology_constants(const unsigned q, const unsigned r, const unsigned s)
    {
        _q = q;
        _r = r;
        _s = s;
    }
    inline std::vector<T> serialize() const
    {
        std::vector<T> out;

        // Serial net dimensions
        out.push_back(static_cast<T>(IN));
        out.push_back(static_cast<T>(OUT));
        out.push_back(static_cast<T>(_nodes.size()));

        // Pack all nodes
        const size_t size = _nodes.size();
        for (size_t i = 0; i < size; i++)
        {
            _nodes[i].serialize(out);
        }

        return out;
    }
    inline void deserialize(const std::vector<T> &data)
    {
        // Use int here, in case someone feeds in garbage data
        // Check input size
        const int in = static_cast<int>(data[0]);
        if (in != IN)
        {
            throw std::runtime_error("nneat: can't deserialize, expected input '" + std::to_string(IN) + "' but got '" + std::to_string(in) + "'");
        }

        // Check output size
        const int out = static_cast<int>(data[1]);
        if (out != OUT)
        {
            throw std::runtime_error("nneat: can't deserialize, expected input '" + std::to_string(OUT) + "' but got '" + std::to_string(out) + "'");
        }

        // Clear the nodes
        _nodes.clear();

        // Read node size
        const int node_size = static_cast<int>(data[2]);
        if (node_size < 0)
        {
            throw std::runtime_error("nneat: can't deserialize, invalid node size");
        }

        // Unpack all nanodes
        _nodes.reserve(node_size);
        size_t start = 3;
        for (int i = 0; i < node_size; i++)
        {
            _nodes.emplace_back(data, start);
        }
    }
};
}
#endif
