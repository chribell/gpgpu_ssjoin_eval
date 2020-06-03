#include <fstream>
#include <sstream>
#include "input.hxx"

void add_input(const std::string & filename, Collection<unsigned int>& collection) {
    std::ifstream infile;
    std::string line;
    infile.open(filename.c_str());

    unsigned int maxElement = 0;
    unsigned int setID = 0;
    unsigned int startSum = 0;

    while (!infile.eof()) {
        std::getline(infile, line);
        if (line == "") continue;

        std::vector<unsigned int> tokens = split(line, ' ');
        unsigned int size = tokens.size();

        if (maxElement < tokens[tokens.size() - 1]) {
            maxElement = tokens[tokens.size() - 1];
        }

        for (unsigned int i = 0; i < size; ++i) {
            collection.addEntry(tokens[i], setID, i);
        }
        collection.addSize(size);
        collection.addStart(startSum);
        startSum += size;
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