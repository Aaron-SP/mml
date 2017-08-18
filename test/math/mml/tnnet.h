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
#ifndef __TEST_NEURAL_NET__
#define __TEST_NEURAL_NET__

#include <mml/nnet.h>
#include <mml/test.h>
#include <mml/vec.h>
#include <stdexcept>

bool test_neural_net()
{
    bool out = true;

    // Test nnet operations
    mml::net_rng<double> rng;
    mml::vector<double, 3> in;
    in[0] = 3.0;
    in[1] = 4.0;
    in[2] = 5.0;
    mml::vector<double, 3> output;
    mml::vector<double, 3> cached_output;
    mml::nnet<double, 3, 3> net;
    mml::nnet<double, 3, 3> net2;
    net.add_layer(5);
    net.add_layer(4);

    // Test net calculation, should be zero
    net.set_input(in);
    output = net.calculate();

    // Test first layer of net
    out = out && compare(0.5, net.get_node(0, 0), 1E-4);
    out = out && compare(0.5, net.get_node(0, 1), 1E-4);
    out = out && compare(0.5, net.get_node(0, 2), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate layer 1");
    }

    // Test second layer of net
    out = out && compare(1.5, net.get_node(1, 0), 1E-4);
    out = out && compare(1.5, net.get_node(1, 1), 1E-4);
    out = out && compare(1.5, net.get_node(1, 2), 1E-4);
    out = out && compare(1.5, net.get_node(1, 3), 1E-4);
    out = out && compare(1.5, net.get_node(1, 4), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate layer 2");
    }

    // Test third layer of net
    out = out && compare(2.5, net.get_node(2, 0), 1E-4);
    out = out && compare(2.5, net.get_node(2, 1), 1E-4);
    out = out && compare(2.5, net.get_node(2, 2), 1E-4);
    out = out && compare(2.5, net.get_node(2, 3), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate layer 3");
    }

    // Test last layer of net
    out = out && compare(2.0, net.get_node(3, 0), 1E-4);
    out = out && compare(2.0, net.get_node(3, 1), 1E-4);
    out = out && compare(2.0, net.get_node(3, 2), 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate layer 4");
    }

    out = out && compare(2.0, output[0], 1E-4);
    out = out && compare(2.0, output[1], 1E-4);
    out = out && compare(2.0, output[2], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate output");
    }

    // Check operator=
    net2 = net;
    output = net2.calculate();
    out = out && compare(2.0, output[0], 1E-4);
    out = out && compare(2.0, output[1], 1E-4);
    out = out && compare(2.0, output[2], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate output copy");
    }

    // Check neural nets are compatible
    const bool compatible = mml::nnet<double, 3, 3>::compatible(net, net2);
    out = out && compatible;
    if (!out)
    {
        throw std::runtime_error("Failed net compatible");
    }

    // Breed the net with itself, output is same since weights are zero
    net2 = mml::nnet<double, 3, 3>::breed(net, net2);
    output = net2.calculate();
    out = out && compare(2.0, output[0], 1E-4);
    out = out && compare(2.0, output[1], 1E-4);
    out = out && compare(2.0, output[2], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate output breed");
    }

    // Try to randomize the net
    cached_output = output;
    net2.randomize(rng);
    output = net2.calculate();
    out = out && !compare(cached_output[0], output[0], 1E-4);
    out = out && !compare(cached_output[1], output[1], 1E-4);
    out = out && !compare(cached_output[2], output[2], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate output random");
    }

    // Test inbreeding
    cached_output = output;
    net2 = mml::nnet<double, 3, 3>::breed(net2, net2);
    output = net2.calculate();
    out = out && compare(cached_output[0], output[0], 1E-4);
    out = out && compare(cached_output[1], output[1], 1E-4);
    out = out && compare(cached_output[2], output[2], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate inbreeding");
    }

    // Mutate the neural net
    cached_output = output;
    net2.mutate(rng);
    output = net2.calculate();
    out = out && !compare(cached_output[0], output[0], 1E-4);
    out = out && !compare(cached_output[1], output[1], 1E-4);
    out = out && !compare(cached_output[2], output[2], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net calculate output random breed mutate");
    }

    // Test input is unchanged
    in = net2.get_input();
    out = out && compare(3.0, in[0], 1E-4);
    out = out && compare(4.0, in[1], 1E-4);
    out = out && compare(5.0, in[2], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net input unchanged");
    }

    // Test serialize neural net
    std::vector<double> data = net2.serialize();
    out = out && compare(37, data.size());
    if (!out)
    {
        throw std::runtime_error("Failed net serialize");
    }

    // Cache old result
    cached_output = output;

    // Test deserialize neural net
    mml::nnet<double, 3, 3> net3;
    net3.deserialize(data);
    net3.set_input(in);
    output = net3.calculate();

    out = out && compare(cached_output[0], output[0], 1E-4);
    out = out && compare(cached_output[1], output[1], 1E-4);
    out = out && compare(cached_output[2], output[2], 1E-4);
    if (!out)
    {
        throw std::runtime_error("Failed net deserialize calculate");
    }

    // return result
    return out;
}

#endif
