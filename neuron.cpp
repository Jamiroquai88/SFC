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


void Neuron::ComputeOutput(std::vector<float> vector, std::vector<float> weights, unsigned int rows, unsigned int cols) {
	std::vector<float> u(rows);
	for (unsigned int i = 0; i < cols; i++) {
		std::vector<float> subvector = Vector::SubVector(vector, i * rows, (i + 1) * rows);
		std::vector<float> subweights = Vector::SubVector(weights, i * rows, (i + 1) * rows);
		u = u + Vector::Dot(subvector, subweights, rows, 1, 1);
	}
//	u = u + m_bias;
	m_output = Vector::Sigmoid(u);
}

void Neuron::Init(unsigned int weight_idx, unsigned int size, float min_w, float max_w) {
	std::vector<float> w;
	m_weights.push_back(w);
	for (unsigned int i = 0; i < size; i++)
		m_weights[weight_idx].push_back(min_w + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_w-min_w))));
}

std::vector<float> Neuron::ConcatWeights() {
	std::vector<float> out;
	for (unsigned int i = 0; i < m_weights.size(); i++)
		for (unsigned int j = 0; j < m_weights[i].size(); j++)
			out.push_back(m_weights[i][j]);
	return out;
}
