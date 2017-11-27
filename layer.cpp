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
	for (unsigned int i = 0; i < m_neurons.size(); i++)
		for (unsigned int j = 0; j < m_neurons[i].m_output.size(); j++)
			out.push_back(m_neurons[i].m_output[j]);
	return out;
}
