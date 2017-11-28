/*
 * neuron.cpp
 *
 *  Created on: Nov 19, 2017
 *      Author: profant
 */

#include <math.h>
#include <cstdlib>
#include <iostream>

#include "vector.h"
#include "neuron.h"


void Neuron::ComputeOutput(std::vector<float> input) {
	if (input.size() != m_weights.size()) {
		std::cerr << "Input size and weights size must be same." << std::endl;
		exit(1);
	}
	float u = 0.0f;
	for (unsigned int i = 0; i < m_weights.size(); i++) {
		u = u + input[i] * m_weights[i];
	}
	m_output = Vector::Sigmoid(u + m_bias);
}

void Neuron::InitWeight(float min_w, float max_w) {
	m_weights.push_back(min_w + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_w-min_w))));
}

std::vector<float> Neuron::ConcatWeights() {
	std::vector<float> out;
	for (unsigned int i = 0; i < m_weights.size(); i++)
		out.push_back(m_weights[i]);
	return out;
}
