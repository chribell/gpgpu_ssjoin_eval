#ifndef BITMAP_GPUSSJOIN_CLASSES_HXX
#define BITMAP_GPUSSJOIN_CLASSES_HXX

#pragma once
#include <numeric>
#include <vector>

#include "definitions.h"

template<class T>
struct Entry{
    T token;
    unsigned int setID;
    Entry() = default;
    Entry(T token, unsigned int setID) : token(token), setID(setID) {}
};

template <class T>
struct Collection
{
    typedef std::vector<T> Tokens;
    typedef std::vector<Entry<T>> Entries;
    typedef std::vector<unsigned int> Cardinalities;
    typedef std::vector<unsigned int> Offsets;
    Tokens tokens;
    Offsets offsets;
    Cardinalities cardinalities;
    Entries entries;
    size_t numberOfTerms = 0;
    Collection() = default;
    Collection(const Collection<T>& collection)
            : tokens(collection.tokens),
              cardinalities(collection.cardinalities),
              offsets(collection.offsets),
              entries(collection.entries),
              numberOfTerms(collection.numberOfTerms) {}

    inline void addToken(T element)
    {
        tokens.push_back(element);
    }
    inline void addOffset(unsigned int offset)
    {
        offsets.push_back(offset);
    }
    inline void addCardinality(unsigned int cardinality)
    {
        cardinalities.push_back(cardinality);
    }
    inline void addEntry(T token, unsigned int setID)
    {
        entries.push_back(Entry<T>(token, setID));
    }
    inline unsigned int numberOfEntriesInRange(unsigned int startID, unsigned int endID)
    {
        return (offsets[endID] + cardinalities[endID]) - (offsets[startID] + cardinalities[startID]);
    }
    inline Tokens at(size_t pos)
    {
        auto first = tokens.begin() + (pos != 0 ? offsets[pos - 1] : 0);
        auto last = first + offsets[pos];
        return Tokens(first, last);
    }
};
#endif //BITMAP_GPUSSJOIN_CLASSES_HXX