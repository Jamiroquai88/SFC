//
// Created by profant on 18.11.17.
//

#include <getopt.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

#include "ivector.h"
#include "backprop.h"


std::string GetDirname(const std::string& path) {
    size_t found = path.find_last_of("/\\");
    return path.substr(0,found);
}


int main(int argc, char *argv[]) {
    int c;
    std::string input_list(""), data_dir("");
    float eps = 0.1f;
    unsigned int size = 250;
    unsigned int test_vectors = 2;
    unsigned int hidden_layer_neurons = 200;

    while ((c = getopt(argc, argv, "l:d:i:t:e:h:")) != -1) {
        switch (c) {
            case 'l':
                input_list = optarg;
                break;
            case 'd':
                data_dir = optarg;
                break;
            case 'i':
                size = atoi(optarg);
                break;
            case 't':
            	test_vectors = atoi(optarg);
            	break;
            case 'e':
            	eps = atof(optarg);
            	break;
            case 'h':
            	hidden_layer_neurons = atoi(optarg);
            	break;
        }
    }

    if (input_list.length() == 0) {
        std::cerr << "Argument -l is required." << std::endl;
        return -1;
    }

    if (data_dir.length() == 0) {
        std::cerr << "Argument -d is required." << std::endl;
        return -1;
    }

    if (size <= 0) {
        std::cerr << "Size must be above zero." << std::endl;
        return -1;
    }

    if (test_vectors < 0) {
    	std::cerr << "Number of test vectors must be above zero." << std::endl;
    	return -1;
    }

    std::ifstream infile(input_list);
    std::string line;
    std::string speaker;

    BackPropagationNN bpnn = BackPropagationNN();

    while (std::getline(infile, line)) {
        speaker = GetDirname(line);
        IVector ivector = IVector(size, speaker);
        if (!ivector.Init(data_dir + '/' + line)) {
            std::cerr << "Failed to read i-vector from " << line << "." << std::endl;
            return -1;
        }
        bpnn.Add(ivector);
    }

    bpnn.Init(test_vectors, hidden_layer_neurons);

    bpnn.Train(eps);

    return 0;
}


