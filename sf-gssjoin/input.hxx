#ifndef SSFGSSJOIN_INPUT_HXX
#define SSFGSSJOIN_INPUT_HXX

#include <cstring>
#include "classes.hxx"


void add_input(const std::string & filename, Collection<unsigned int>& collection);
std::vector<unsigned int> split(const std::string& s, char delim);
#endif // SSFGSSJOIN_INPUT_HXX
