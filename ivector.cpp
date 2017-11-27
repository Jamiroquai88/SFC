//
// Created by profant on 18.11.17.
//

#include "ivector.h"

IVector::IVector(int size, std::string speaker) {
    this->m_size = size;
    this->m_speaker = speaker;
}

bool IVector::Init(std::string filename) {
    std::ifstream file(filename, std::ifstream::in);
    if (file) {
        float myf;
        for (unsigned int i = 0; i < m_size; i++) {
            file >> myf;
            this->m_data.push_back(myf);
        }
        return true;
    }
    else {
        return false;
    }
}
