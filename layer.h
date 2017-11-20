/*
 * layer.h
 *
 *  Created on: Nov 19, 2017
 *      Author: profant
 */

#include "neuron.h"


#ifndef LAYER_H_
#define LAYER_H_

class Layer {
public:
	Neuron * AddNeuron();

	std::vector<Neuron> m_neurons;
};

#endif /* LAYER_H_ */
