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
	return &m_neurons.back();
}


