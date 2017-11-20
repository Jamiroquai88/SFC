//
// Created by profant on 18.11.17.
//

#include <vector>
#include <set>

#include "ivector.h"
#include "neuron.h"
#include "layer.h"


#ifndef SFC_BACKPROP_H
#define SFC_BACKPROP_H


class BackPropagationNN {
public:

    BackPropagationNN() : m_size(0) {};

    void Add(const IVector& ivec);
    std::vector<IVector> GetSpeakerData(std::string speaker);
    void Init(unsigned int test_ivectors, unsigned int hidden_layer_neurons);
    void Train();

    static std::vector<float> Vectors2Matrix(const std::vector<IVector>& vectors);

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
