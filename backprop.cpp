//
// Created by profant on 18.11.17.
//

#include <algorithm>
#include <unordered_map>
#include <cstdlib>
#include <ctime>

#include "backprop.h"
#include "vector.h"


void BackPropagationNN::Add(const IVector& ivec) {
	m_vectorSize = ivec.m_size;
    m_ivectors.push_back(ivec);
    if (std::find(m_speakers.begin(), m_speakers.end(), ivec.m_speaker) == m_speakers.end()) {
    	m_speakers.push_back(ivec.m_speaker);
    	m_mapping[ivec.m_speaker] = m_speakers.size();
    }
}

std::vector<IVector> BackPropagationNN::GetSpeakerData(std::string speaker) {
	std::vector<IVector> speaker_data;
	for (auto i : m_ivectors) {
		if (speaker == i.m_speaker)
			speaker_data.push_back(i);
	}
	return speaker_data;
}

void BackPropagationNN::Init(unsigned int num_test_ivectors, unsigned int hidden_layer_neurons) {
	std::unordered_map<std::string, std::vector<IVector>> ivec_map;
	for (auto spk : m_speakers)
		ivec_map[spk] = GetSpeakerData(spk);
	for (auto i : ivec_map) {
		unsigned int size = i.second.size();
		for (unsigned int j = 0; j < size; j++) {
			if (j < num_test_ivectors)
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
		for (unsigned int j = 0; j < m_vectorSize; j++)
			n->InitWeight(min_w, max_w);
	}

	// Output layer - initialize neurons and weights
	for (unsigned int i = 0; i < m_speakers.size(); i++) {
		Neuron * n = m_outputLayer.AddNeuron();
		for (unsigned int j = 0; j < hidden_layer_neurons; j++)
			n->InitWeight(min_w, max_w);
	}
}

void BackPropagationNN::Train(float eps) {
	unsigned int epoch = 1;
	float best_accuracy = 0.0f;
	do {

		// for all training i-vectors
		for (unsigned int i = 0; i < m_trainIVectors.size(); i++) {
			// Get output - forward pass
			std::vector<float> prediction = ForwardPass(m_trainIVectors[i].m_data);

			// Get labels
			std::vector<float> labels = GetLabels(m_trainIVectors[i]);

			// Compute error
			std::vector<float> prediction_error = prediction - labels;

			std::vector<float> hiddenl_weights;

			// Output layer update
			for (unsigned int j = 0; j < m_outputLayer.m_neurons.size(); j++)
				for (unsigned int k = 0; k < m_outputLayer.m_neurons[j].m_weights.size(); k++)
					hiddenl_weights.push_back(m_outputLayer.m_neurons[j].m_weights[k] - eps *
					(prediction_error[j] * (prediction[j] * (1.0f - prediction[j])) * m_hiddenLayer.m_neurons[k].m_output));

			// Hidden layer update
			for (unsigned int j = 0; j < m_hiddenLayer.m_neurons.size(); j++) {
				float sum = 0.0f;
				for (unsigned int k = 0; k < m_outputLayer.m_neurons.size(); k++)
					sum = sum + prediction_error[k] * (prediction[k] * (1.0f - prediction[k])) * m_outputLayer.m_neurons[k].m_weights[j];
				for (unsigned int k = 0; k < m_outputLayer.m_neurons.size(); k++) {
					float comp = (sum * (m_hiddenLayer.m_neurons[j].m_output -
							m_hiddenLayer.m_neurons[j].m_output * m_hiddenLayer.m_neurons[j].m_output)) * m_trainIVectors[i].m_data[k];
					m_hiddenLayer.m_neurons[j].m_weights[k] = m_hiddenLayer.m_neurons[j].m_weights[k] - eps * comp;
				}
			}

			// Assign updated weights
			for (unsigned int i = 0; i < m_outputLayer.m_neurons.size(); i++)
				for (unsigned int j = 0; j < m_outputLayer.m_neurons[i].m_weights.size(); j++)
					m_outputLayer.m_neurons[i].m_weights[j] = hiddenl_weights[i * m_outputLayer.m_neurons[i].m_weights.size() + j];
		}
		// test on testing data
		float accuracy = Test();
		if (accuracy > best_accuracy) {
			best_accuracy = accuracy;
			std::cout << "Epoch: " << epoch << ", Test Accuracy: " <<  best_accuracy << "%" << std::endl;
		}
		epoch++;
	} while (true);
}

std::vector<float> BackPropagationNN::Vectors2Matrix(const std::vector<IVector>& vectors) {
	std::vector<float> out;
	for (unsigned int i = 0; i < vectors.size(); i++)
		for (unsigned int j = 0; j < vectors[i].m_size; j++)
			out.push_back(vectors[i].m_data[j]);
	return out;
}

std::vector<float> BackPropagationNN::GetLabels(IVector& ivector) {
	std::vector<float> out;
	for (unsigned int i = 0; i < m_outputLayer.m_size; i++)
		if (ivector.m_speaker == m_speakers[i])
			out.push_back(1.0f);
		else
			out.push_back(0.0f);
	return out;
}

float BackPropagationNN::GetTotalError(const std::vector<float>& prediction, const std::vector<float>& labels) {
	float error = 0.0f;
	for (unsigned int i = 0; i < prediction.size(); i++)
		error += 0.5f * ((labels[i] - prediction[i]) * (labels[i] - prediction[i]));
	return error;
}

float BackPropagationNN::Test() {
	std::vector<float> prediction;
	unsigned int class_index;
	unsigned int hit = 0;
	for (unsigned int i = 0; i < m_testIVectors.size(); i++) {
		 prediction = ForwardPass(m_testIVectors[i].m_data);
		 class_index = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
		 if (m_speakers[class_index] == m_testIVectors[i].m_speaker)
			 hit++;
	}
	return (float)hit / (float)m_testIVectors.size() * 100;
}

std::vector<float> BackPropagationNN::ForwardPass(std::vector<float> vector) {
	// Hidden layer
	for (unsigned int j = 0; j < m_hiddenLayer.m_neurons.size(); j++)
		m_hiddenLayer.m_neurons[j].ComputeOutput(vector);

	// Output of hidden layer
	std::vector<float> hiddenl_output = m_hiddenLayer.ConcatOutputs();

	// Output layer
	for (unsigned int j = 0; j < m_outputLayer.m_neurons.size(); j++)
		m_outputLayer.m_neurons[j].ComputeOutput(hiddenl_output);
	return m_outputLayer.ConcatOutputs();
}
