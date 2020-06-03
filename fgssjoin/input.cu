#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <iostream>
#include "input.hxx"
#include "similarity.hxx"

void add_input(const std::string & filename, Collection<unsigned int>& collection, double threshold, bool binaryJoin) {
    std::ifstream infile;
    std::string line;
    infile.open(filename.c_str());

    unsigned int maxElement = 0;
    unsigned int setID = 0;
    unsigned int startSum = 0;
    unsigned int prefixStartSum = 0;

    while (!infile.eof()) {
        std::getline(infile, line);
        if (line == "") continue;

        std::vector<unsigned int> tokens = split(line, ' ');
        unsigned int size = tokens.size();

        if (maxElement < tokens[tokens.size() - 1]) {
            maxElement = tokens[tokens.size() - 1];
        }

        unsigned int prefix = binaryJoin ? jaccard_maxprefix(size, threshold) : jaccard_midprefix(size, threshold);

        for (unsigned int i = 0; i < size; ++i) {
            collection.addEntry(tokens[i], setID, i);
            if (i < prefix) {
                collection.addPrefixEntry(tokens[i], setID, i);
            }
        }
        collection.addSize(size);
        collection.addPrefixStart(prefixStartSum);
        collection.addStart(startSum);
        startSum += size;
        prefixStartSum += prefix;
        setID++;
    }
    collection.universeSize = maxElement + 1; // add one for the zero token
    infile.close();
}

std::vector<unsigned int> split(const std::string& s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<unsigned int> elements;
    while (std::getline(ss, item, delim)) {
        elements.push_back(atoi(item.c_str()));
    }
    return elements;
}