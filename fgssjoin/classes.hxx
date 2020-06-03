#ifndef FSSJOIN_CLASSES_HXX
#define FSSJOIN_CLASSES_HXX

#pragma once
#include <numeric>
#include <vector>

template<class T>
struct Entry{
    T token;
    unsigned int setID;
    unsigned int position;
    Entry() = default;
    Entry(T token, unsigned int setID, unsigned int position) : token(token), setID(setID), position(position) {}
};

template <class T>
struct Collection
{
    typedef std::vector<Entry<T>> Entries;
    typedef std::vector<Entry<T>> PrefixEntries;
    typedef std::vector<unsigned int> Sizes;
    typedef std::vector<unsigned int> Starts;
    typedef std::vector<unsigned int> PrefixStarts;
    Entries entries;
    PrefixEntries prefixEntries;
    Sizes sizes;
    Starts starts;
    PrefixStarts prefixStarts;
    size_t universeSize = 0;
    Collection() = default;
    Collection(const Collection<T>& collection)
            : entries(collection.entries),
              prefixEntries(collection.prefixEntries),
              sizes(collection.sizes),
              starts(collection.starts),
              prefixStarts(collection.prefixStarts),
              universeSize(collection.universeSize) {}

    inline void addStart(unsigned int start)
    {
        starts.push_back(start);
    }
    inline void addSize(unsigned int size)
    {
        sizes.push_back(size);
    }
    inline void addPrefixStart(unsigned int start)
    {
        prefixStarts.push_back(start);
    }
    inline void addEntry(T token, unsigned int setID, unsigned int position)
    {
        entries.push_back(Entry<T>(token, setID, position));
    }
    inline void addPrefixEntry(T token, unsigned int setID, unsigned int position)
    {
        prefixEntries.push_back(Entry<T>(token, setID, position));
    }
    inline Entries at(size_t pos)
    {
        auto first = entries.begin() + (pos != 0 ? starts[pos - 1] : 0);
        auto last = first + starts[pos];
        return Entries(first, last);
    }
};
#endif // FSSJOIN_CLASSES_HXX