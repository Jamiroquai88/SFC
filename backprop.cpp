//
// Created by profant on 18.11.17.
//

#include <unordered_map>
#include <cstdlib>
#include <ctime>

#include "backprop.h"
#include "vector.h"


void BackPropagationNN::Add(const IVector& ivec) {
	m_size = ivec.m_size;
    m_ivectors.push_back(ivec);
    m_speakers.insert(ivec.m_speaker);
}

std::vector<IVector> BackPropagationNN::GetSpeakerData(std::string speaker) {
	std::vector<IVector> speaker_data;
	for (auto i : m_ivectors) {
		if (speaker == i.m_speaker)
			speaker_data.push_back(i);
	}
	return speaker_data;
}

void BackPropagationNN::Init(unsigned int test_ivectors, unsigned int hidden_layer_neurons) {
	std::unordered_map<std::string, std::vector<IVector>> ivec_map;
	for (auto spk : m_speakers)
		ivec_map[spk] = GetSpeakerData(spk);
	for (auto i : ivec_map) {
		unsigned int size = i.second.size();
		for (unsigned int j = 0; j < size; j++) {
			if (j < test_ivectors)
				m_testIVectors.push_back(i.second[j]);
			else
				m_trainIVectors.push_back(i.second[j]);
		}
	}
	std::cout << "Test ivectors: " << m_testIVectors.size() << ", train ivectors: " << m_trainIVectors.size() << std::endl;

	// Initialize rand
	srand (static_cast <unsigned> (time(0)));

	// Min and max weight initialization
	float min_w = -0.5;
	float max_w = 0.5;

	// Hidden layer - initialize neurons and weights
	for (unsigned int i = 0; i < hidden_layer_neurons; i++) {
		Neuron * n = m_hiddenLayer.AddNeuron();
		for (unsigned int j = 0; j < m_speakers.size(); j++)
			n->Init(j, m_size, min_w, max_w);
	}

	// Output layer
	for (unsigned int i = 0; i < m_speakers.size(); i++) {
		Neuron * n = m_outputLayer.AddNeuron();
		for (unsigned int j = 0; j < hidden_layer_neurons; j++)
			n->Init(j, m_size, min_w, max_w);
	}
}

void BackPropagationNN::Train() {
	std::vector<float> train_vectors = Vectors2Matrix(m_trainIVectors);
	unsigned int rows = m_trainIVectors[0].m_size;
	unsigned int train_cols = m_trainIVectors.size();
	do {
		std::vector<float> input_sig = Vector::Sigmoid(train_vectors);

		// Hidden layer
		for (unsigned int i = 0; i < m_hiddenLayer.m_neurons.size(); i++)
			m_hiddenLayer.m_neurons[i].ComputeOutput(input_sig, rows, train_cols);

		// Output layer
		std::vector<float> con_out = m_hiddenLayer.ConcatOutputs();
		for (auto on : m_outputLayer.m_neurons)
				on.ComputeOutput(con_out, rows, m_hiddenLayer.m_size);
	} while (false);
}

std::vector<float> BackPropagationNN::Vectors2Matrix(const std::vector<IVector>& vectors) {
	std::vector<float> out;
	for (auto vector : vectors)
		for (auto elem : vector.m_data)
			out.push_back(elem);
	return out;
}
