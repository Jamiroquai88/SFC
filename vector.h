/*
 * vector.h
 *
 *  Created on: Nov 19, 2017
 *      Author: profant
 */

#include <vector>

#ifndef VECTOR_H_
#define VECTOR_H_


class Vector {
public:

    static std::vector<float> Dot(const std::vector <float>& m1, const std::vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns);
    static std::vector<float> Transpose (float *m, const int C, const int R);
    static std::vector<float> Sigmoid (const std::vector<float>& m1);
    static float              Sigmoid (const float& m1);
    static std::vector<float> SigmoidDer(const std::vector<float>& m1);
    static std::vector<float> SubVector(const std::vector<float>& vec, unsigned int start, unsigned int end);
};

inline std::vector<float> operator+(const std::vector <float>& m1, const std::vector <float>& m2) {

     /*  Returns the sum of two vectors.
         Inputs:
             m1: vector
             m2: vector
         Output: vector, m1 + m2, difference between two vectors m1 and m2.
     */

     const unsigned long VECTOR_SIZE = m1.size();
     std::vector <float> sum (VECTOR_SIZE);

     for (unsigned i = 0; i != VECTOR_SIZE; ++i){
     	sum[i] = m1[i] + m2[i];
     };

     return sum;
}

inline std::vector<float> operator-(const std::vector <float>& m1, const std::vector <float>& m2) {

     /*  Returns the difference between two vectors.
         Inputs:
             m1: vector
             m2: vector
         Output: vector, m1 - m2, difference between two vectors m1 and m2.
     */

     const unsigned long VECTOR_SIZE = m1.size();
     if (VECTOR_SIZE != m2.size()) {
    	 std::cerr << "Invalid number of dimensions. Vector 1: " << VECTOR_SIZE << ", Vector 2: " << m2.size() << std::endl;
    	 exit(-1);
     }

     std::vector <float> difference (VECTOR_SIZE);

     for (unsigned i = 0; i != VECTOR_SIZE; ++i){
         difference[i] = m1[i] - m2[i];
     };

     return difference;
}

inline std::vector <float> operator*(const std::vector <float>& m1, const std::vector <float>& m2){

	/*  Returns the product of two vectors (elementwise multiplication).
		Inputs:
			m1: vector
			m2: vector
		Output: vector, m1 * m2, product of two vectors m1 and m2
	*/

	const unsigned long VECTOR_SIZE = m1.size();
	std::vector <float> product (VECTOR_SIZE);

	for (unsigned i = 0; i != VECTOR_SIZE; ++i){
		product[i] = m1[i] * m2[i];
	};

	return product;
}

inline std::vector <float> operator*(float m1, const std::vector <float>& m2) {
	const unsigned long VECTOR_SIZE = m2.size();
	std::vector <float> product (VECTOR_SIZE);

	for (unsigned i = 0; i != VECTOR_SIZE; ++i)
	    product[i] = m1 * m2[i];

	return product;
}

inline std::vector <float> operator*(const std::vector <float>& m2, float m1) {
	return m1 * m2;
}

#endif /* VECTOR_H_ */
