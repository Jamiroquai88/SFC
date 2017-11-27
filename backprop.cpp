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
		for (unsigned int j = 0; j < m_trainIVectors.size(); j++)
			n->Init(j, m_vectorSize, min_w, max_w);
	}

	// Output layer - initialize neurons and weights
	for (unsigned int i = 0; i < m_trainIVectors.size(); i++) {
		Neuron * n = m_outputLayer.AddNeuron();
		for (unsigned int j = 0; j < hidden_layer_neurons; j++)
			n->Init(j, m_vectorSize, min_w, max_w);
	}

	// Initialize hard coded
//	m_hiddenLayer.m_neurons[0].m_weights[0] = {0.15};
//	m_hiddenLayer.m_neurons[0].m_weights[1] = {0.20};
//	m_hiddenLayer.m_neurons[0].m_bias = {0.35};
//	m_hiddenLayer.m_neurons[1].m_weights[0] = {0.25};
//	m_hiddenLayer.m_neurons[1].m_weights[1] = {0.30};
//	m_hiddenLayer.m_neurons[1].m_bias = {0.35};
//	m_outputLayer.m_neurons[0].m_weights[0] = {0.40};
//	m_outputLayer.m_neurons[0].m_weights[1] = {0.45};
//	m_outputLayer.m_neurons[0].m_bias = {0.60};
//	m_outputLayer.m_neurons[1].m_weights[0] = {0.50};
//	m_outputLayer.m_neurons[1].m_weights[1] = {0.55};
//	m_outputLayer.m_neurons[1].m_bias = {0.60};
}

void BackPropagationNN::Train(float eps) {
	std::vector<float> train_vectors = Vector::Sigmoid(Vectors2Matrix(m_trainIVectors));
	unsigned int rows = m_vectorSize;
	unsigned int train_cols = m_trainIVectors.size();
	std::vector<float> labels = GetMapping();
	do {
		// Hidden layer
		for (unsigned int i = 0; i < m_hiddenLayer.m_neurons.size(); i++)
			m_hiddenLayer.m_neurons[i].ComputeOutput(train_vectors, m_hiddenLayer.m_neurons[i].ConcatWeights(), rows, train_cols);

		// Output of hidden layer
		std::vector<float> hiddenl_output = m_hiddenLayer.ConcatOutputs();

		std::cout << "Size of hidden layer output: " << hiddenl_output.size() << std::endl;

		// Output layer
		for (unsigned int i = 0; i < m_outputLayer.m_neurons.size(); i++)
			m_outputLayer.m_neurons[i].ComputeOutput(hiddenl_output, m_outputLayer.m_neurons[i].ConcatWeights(), rows, m_hiddenLayer.m_size);

		// Get output
		std::vector<float> prediction = m_outputLayer.ConcatOutputs();

//		std::vector<float> tmp = Vector::Dot(prediction, )

		// Compute error
		float total_error = GetTotalError(prediction, labels);
		std::cout << "Total Prediction Error: " << total_error << std::endl;
		std::vector<float> prediction_error = prediction - labels;

		std::vector<std::vector<float>> hiddenl_weights;

		// Output layer update
		for (unsigned int i = 0; i < m_outputLayer.m_neurons.size(); i++)
			for (unsigned int j = 0; j < m_outputLayer.m_neurons[i].m_weights.size(); j++)
				hiddenl_weights.push_back(m_outputLayer.m_neurons[i].m_weights[j] - eps *
				(prediction_error[i] * (prediction[i] * (1.0f - prediction[i])) * m_hiddenLayer.m_neurons[j].m_output));

		// Hidden layer update
		for (unsigned int i = 0; i < m_hiddenLayer.m_neurons.size(); i++) {
			std::vector<float> sum(rows);
			for (unsigned int j = 0; j < m_outputLayer.m_neurons.size(); j++)
				sum = sum + prediction_error[j] * (prediction[j] * (1.0f - prediction[j])) * m_outputLayer.m_neurons[j].m_weights[i];
			for (unsigned int j = 0; j < m_outputLayer.m_neurons.size(); j++) {
				std::vector<float> comp = (sum * (m_hiddenLayer.m_neurons[i].m_output -
						m_hiddenLayer.m_neurons[i].m_output * m_hiddenLayer.m_neurons[i].m_output)) * train_vectors[j];
				m_hiddenLayer.m_neurons[i].m_weights[j] = m_hiddenLayer.m_neurons[i].m_weights[j] - eps * comp;
			}
		}

		// Assign updated weights
		for (unsigned int i = 0; i < m_outputLayer.m_neurons.size(); i++)
			for (unsigned int j = 0; j < m_outputLayer.m_neurons[i].m_weights.size(); j++)
				m_outputLayer.m_neurons[i].m_weights[j] = hiddenl_weights[i * m_outputLayer.m_neurons[i].m_weights.size() + j];

//		Test();

	} while (true);
}

std::vector<float> BackPropagationNN::Vectors2Matrix(const std::vector<IVector>& vectors) {
	std::vector<float> out;
	for (unsigned int i = 0; i < vectors.size(); i++)
		for (unsigned int j = 0; j < vectors[i].m_size; j++)
			out.push_back(vectors[i].m_data[j]);
	return out;
}

std::vector<float> BackPropagationNN::GetMapping() {
	std::vector<float> out;
	unsigned int size = m_trainIVectors.size();
	for (unsigned int i = 0; i < size; i++)
		out.push_back((float)m_mapping[m_trainIVectors[i].m_speaker] / (float)size);
	return out;
}

float BackPropagationNN::GetTotalError(const std::vector<float>& prediction, const std::vector<float>& labels) {
	float error = 0.0f;
	for (unsigned int i = 0; i < prediction.size(); i++)
		error += 0.5f * ((labels[i] - prediction[i]) * (labels[i] - prediction[i]));
	return error;
}

void BackPropagationNN::Test() {
	std::vector<float> test_vectors = Vector::Sigmoid(Vectors2Matrix(m_testIVectors));
	unsigned int rows = m_vectorSize;
	unsigned int train_cols = m_trainIVectors.size();

	for (unsigned int i = 0; i < m_hiddenLayer.m_neurons.size(); i++)
		m_hiddenLayer.m_neurons[i].ComputeOutput(test_vectors, m_hiddenLayer.m_neurons[i].ConcatWeights(), rows, train_cols);

	// Output of hidden layer
	std::vector<float> hiddenl_output = m_hiddenLayer.ConcatOutputs();

	// Output layer
	for (unsigned int i = 0; i < m_outputLayer.m_neurons.size(); i++)
		m_outputLayer.m_neurons[i].ComputeOutput(hiddenl_output, m_outputLayer.m_neurons[i].ConcatWeights(), rows, m_hiddenLayer.m_size);

	// Get output
	std::vector<float> prediction = m_outputLayer.ConcatOutputs();
}
