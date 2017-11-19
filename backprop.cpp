//
// Created by profant on 18.11.17.
//

#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <math.h>

#include "backprop.h"


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
		Neuron n = m_hiddenLayer.AddNeuron();
		for (unsigned int j = 0; j < m_speakers.size(); j++)
			n.Init(j, m_size, min_w, max_w);
	}

	// Output layer
	for (unsigned int i = 0; i < m_speakers.size(); i++) {
		Neuron n = m_outputLayer.AddNeuron();
		for (unsigned int j = 0; j < hidden_layer_neurons; j++)
			n.Init(j, m_size, min_w, max_w);
	}
}

void BackPropagationNN::Train() {
	do {
		for (auto trn_ivec : m_trainIVectors) {
			// Input layer
			std::vector<float> input_sig = Sigmoid(trn_ivec.m_data);

			// Hidden layer
			for (auto neuron : m_hiddenLayer.m_neurons) {

			}
		}
	} while (true);
}

Neuron Layer::AddNeuron() {
	Neuron n = Neuron();
	m_neurons.push_back(n);
	return n;
}

void Neuron::Init(unsigned int weight, unsigned int size, float min_w, float max_w) {
	std::vector<float> w;
	m_weights.push_back(w);
	std::cout << "Initializing neuron weight " << weight + 1 << " from " << m_weights.size() << std::endl;
	for (unsigned int i = 0; i < size; i++)
		m_weights[weight].push_back(min_w + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_w-min_w))));
}

std::vector<float> BackPropagationNN::Dot(const std::vector <float>& m1, const std::vector <float>& m2,
                    const int m1_rows, const int m1_columns, const int m2_columns) {

    /*  Returns the product of two matrices: m1 x m2.
        Inputs:
            m1: vector, left matrix of size m1_rows x m1_columns
            m2: vector, right matrix of size m1_columns x m2_columns
                (the number of rows in the right matrix must be equal
                to the number of the columns in the left one)
            m1_rows: int, number of rows in the left matrix m1
            m1_columns: int, number of columns in the left matrix m1
            m2_columns: int, number of columns in the right matrix m2
        Output: vector, m1 * m2, product of two vectors m1 and m2,
                a matrix of size m1_rows x m2_columns
    */

    std::vector <float> output (m1_rows*m2_columns);

    for( int row = 0; row != m1_rows; ++row ) {
        for( int col = 0; col != m2_columns; ++col ) {
            output[ row * m2_columns + col ] = 0.f;
            for( int k = 0; k != m1_columns; ++k ) {
                output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
            }
        }
    }

    return output;
}

std::vector<float> BackPropagationNN::Sigmoid (const std::vector <float>& m1) {

    /*  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: 1/(1 + e^-x) for every element of the input matrix m1.
    */

    const unsigned long VECTOR_SIZE = m1.size();
    std::vector<float> output (VECTOR_SIZE);


    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = 1 / (1 + exp(-m1[ i ]));
    }

    return output;
}
