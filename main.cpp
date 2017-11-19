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
    int size = 250;

    while ((c = getopt(argc, argv, "l:d:i:")) != -1) {
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

    std::ifstream infile(input_list);
    std::string line;
    std::string speaker;

    BackPropagationNN bpnn = BackPropagationNN();

    while (std::getline(infile, line)) {
        speaker = GetDirname(line);
        std::cout << line << " " << speaker << std::endl;
        IVector ivector = IVector(size, speaker);
        if (!ivector.Init(data_dir + '/' + line)) {
            std::cerr << "Failed to read i-vector from " << line << "." << std::endl;
            return -1;
        }
        bpnn.Add(ivector);
    }



        return 0;
}

