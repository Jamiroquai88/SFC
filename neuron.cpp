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


void Neuron::ComputeOutput(std::vector<float> vector) {
	unsigned int size = vector.size();
	std::vector<float> u(size);
	for (unsigned int i = 0; i < m_weights.size(); i++)
		u = Vector::Add(u, Vector::Dot(vector, m_weights[i], size, size, 1));
	m_output = Vector::Sigmoid(u);
}

void Neuron::Init(unsigned int weight, unsigned int size, float min_w, float max_w) {
	std::vector<float> w;
	m_weights.push_back(w);
	std::vector<float> output;
	std::cout << "Initializing neuron weight " << weight + 1 << " from " << m_weights.size() << std::endl;
	for (unsigned int i = 0; i < size; i++)
		m_weights[weight].push_back(min_w + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_w-min_w))));
}


