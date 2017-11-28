/*
 * vector.cpp
 *
 *  Created on: Nov 19, 2017
 *      Author: profant
 */

#include <math.h>
#include <iostream>

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

std::vector<float> Vector::SigmoidDer(const std::vector <float>& m1) {

    /*  Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
        where f(x) is sigmoid function.
        Input: m1, a vector.
        Output: x(1 - x) for every element of the input matrix m1.
    */

    const unsigned long VECTOR_SIZE = m1.size();
    std::vector<float> output (VECTOR_SIZE);

    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = m1[ i ] * (1 - m1[ i ]);
    }

    return output;
}

std::vector <float> Vector::Transpose (float *m, const int C, const int R) {

     /*  Returns a transpose matrix of input matrix.
         Inputs:
             m: vector, input matrix
             C: int, number of columns in the input matrix
             R: int, number of rows in the input matrix
         Output: vector, transpose matrix mT of input matrix m
     */

     std::vector <float> mT (C*R);

     for(int n = 0; n!=C*R; n++) {
         int i = n/C;
         int j = n%C;
         mT[n] = m[R*j + i];
     }

     return mT;
}

std::vector<float> Vector::SubVector(const std::vector<float>& vec, unsigned int start, unsigned int end) {
	std::vector<float>::const_iterator first = vec.begin() + start;
	std::vector<float>::const_iterator last = vec.begin() + end;
	std::vector<float> newVec(first, last);
	return newVec;
}

float Vector::Sigmoid(const float& m1) {
	return 1 / (1 + exp(-m1));
}
