//
// Created by profant on 18.11.17.
//

#include <vector>

#include "ivector.h"


#ifndef SFC_BACKPROP_H
#define SFC_BACKPROP_H

class BackPropagationNN {
public:

    BackPropagationNN() {};

    void Add(const IVector& ivec);
    void Split();

    std::vector<IVector> m_ivectors;
};

#endif //SFC_BACKPROP_H
