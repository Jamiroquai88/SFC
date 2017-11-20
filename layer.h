/*
 * layer.h
 *
 *  Created on: Nov 19, 2017
 *      Author: profant
 */

#include <vector>

#include "neuron.h"


#ifndef LAYER_H_
#define LAYER_H_

class Layer {
public:
	Neuron * AddNeuron();
	std::vector<float> ConcatOutputs();

	std::vector<Neuron> m_neurons;
	unsigned int m_size;
};

#endif /* LAYER_H_ */
