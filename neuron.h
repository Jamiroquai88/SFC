/*
 * neuron.h
 *
 *  Created on: Nov 19, 2017
 *      Author: profant
 */

#include <vector>

#ifndef NEURON_H_
#define NEURON_H_

class Neuron {
public:
	void Init(unsigned int weight, unsigned int size, float min_w, float max_w);
	void ComputeOutput(std::vector<float> vector, unsigned int rows, unsigned int cols);

	std::vector<std::vector<float>> m_weights;
	std::vector<float> m_output;
};

#endif /* NEURON_H_ */
