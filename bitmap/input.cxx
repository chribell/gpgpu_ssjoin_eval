#include <fstream>
#include <unordered_set>
#include <algorithm>
#include "input.hxx"

void add_input(const std::string & filename, Collection<unsigned int>& collection) {
    std::ifstream infile;
    std::string line;
    infile.open(filename.c_str());
    std::unordered_set<unsigned int> s;
    unsigned int id = 0;
    if(infile.is_open()) {
        tokenize_whitespace tw;
        unsigned int globalOffset = 0;
        while(getline(infile, line)) {
            tw.setline(line.c_str());
            unsigned int size = 0;
            while(!tw.end()) {
                const char * token = tw.next();
                unsigned int nmb = atoi(token);
                collection.addToken(nmb);
                collection.addEntry(nmb, id);
                s.insert(nmb);
                size++;
            }
            collection.addCardinality(size);
            collection.addOffset(globalOffset);
            globalOffset += size;
            id++;
        }
    } else {
        perror("could not open file");
        exit(7);
    }
    collection.numberOfTerms = s.size() + 1; // add one for the zero token
    infile.close();
}