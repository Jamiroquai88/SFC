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

    static std::vector<float> Dot(const std::vector <float>& m1, const std::vector <float>& m2,
                        const int m1_rows, const int m1_columns, const int m2_columns);
    static std::vector<float> Sigmoid (const std::vector <float>& m1);
    static std::vector<float> Add(const std::vector <float>& m1, const std::vector <float>& m2);
};



#endif /* VECTOR_H_ */
