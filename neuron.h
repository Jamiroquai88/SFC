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
	void Init(unsigned int, unsigned int, float, float);
	void ComputeOutput(std::vector<float>, std::vector<float>, unsigned int, unsigned int);
	std::vector<float> ConcatWeights();

	std::vector<std::vector<float>> m_weights;
	std::vector<float> m_output;
	std::vector<float> m_input;
//	std::vector<float> m_bias;
};

#endif /* NEURON_H_ */
