#ifndef FGSSJOIN_INPUT_HXX
#define FGSSJOIN_INPUT_HXX

#include <cstring>
#include "classes.hxx"

#define LINEBUFSIZE 8192

void add_input(const std::string & filename, Collection<unsigned int>& collection, double threshold, bool binaryJoin);
std::vector<unsigned int> split(const std::string& s, char delim);
#endif // FGSSJOIN_INPUT_HXX
