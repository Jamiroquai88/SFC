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
	void InitWeight(float, float);
	void ComputeOutput(std::vector<float>);
	std::vector<float> ConcatWeights();

	std::vector<float> m_weights;
	float m_output;
	float m_input;
	float m_bias;
};

#endif /* NEURON_H_ */
