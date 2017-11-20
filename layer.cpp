/*
 * layer.cpp
 *
 *  Created on: Nov 19, 2017
 *      Author: profant
 */

#include "layer.h"


Neuron * Layer::AddNeuron() {
	Neuron n = Neuron();
	m_neurons.push_back(n);
	m_size = m_neurons.size();
	return &m_neurons.back();
}

std::vector<float> Layer::ConcatOutputs() {
	std::vector<float> out;
	for (auto neuron : m_neurons)
		for (auto val : neuron.m_output)
			out.push_back(val);
	return out;
}
