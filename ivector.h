//
// Created by profant on 18.11.17.
//

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#ifndef SFC_IVECTOR_H
#define SFC_IVECTOR_H

class IVector {
public:
    IVector(int size, std::string speaker);

    bool Init(std::string filename);

    int m_size;
    std::vector<float> m_data;
    std::string m_speaker;
};

#endif //SFC_IVECTOR_H
