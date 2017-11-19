/*
 * vector.cpp
 *
 *  Created on: Nov 19, 2017
 *      Author: profant
 */

#include <math.h>

#include "vector.h"


std::vector<float> Vector::Dot(const std::vector <float>& m1, const std::vector <float>& m2,
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

std::vector<float> Vector::Sigmoid(const std::vector <float>& m1) {

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

std::vector<float> Vector::Add(const std::vector <float>& m1, const std::vector <float>& m2) {
	unsigned int size = m1.size();
	std::vector<float> output(size);
	for (unsigned int i = 0; i < size; i++)
		output[i] = m1[i] + m2[i];
	return output;
}



