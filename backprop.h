//
// Created by profant on 18.11.17.
//

#include <vector>
#include <set>

#include "ivector.h"


#ifndef SFC_BACKPROP_H
#define SFC_BACKPROP_H


class Neuron {
public:
	void Init(unsigned int weight, unsigned int size, float min_w, float max_w);

	std::vector<std::vector<float>> m_weights;
};

class Layer {
public:
	Neuron AddNeuron();

	std::vector<Neuron> m_neurons;
};

class BackPropagationNN {
public:

    BackPropagationNN() : m_size(0) {};

    void Add(const IVector& ivec);
    std::vector<IVector> GetSpeakerData(std::string speaker);
    void Init(unsigned int test_ivectors, unsigned int hidden_layer_neurons);
    void Train();

    std::vector<float> Dot(const std::vector <float>& m1, const std::vector <float>& m2,
                        const int m1_rows, const int m1_columns, const int m2_columns);
    std::vector<float> Sigmoid (const std::vector <float>& m1);

    unsigned int m_size;
    std::vector<IVector> m_ivectors;
    std::vector<IVector> m_testIVectors;
    std::vector<IVector> m_trainIVectors;

    std::set<std::string> m_speakers;

    Layer m_inputLayer;
    Layer m_hiddenLayer;
    Layer m_outputLayer;
};

#endif //SFC_BACKPROP_H
