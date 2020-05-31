#ifndef BITMAP_GPUSSJOIN_DEFINITIONS_HXX
#define BITMAP_GPUSSJOIN_DEFINITIONS_HXX

#include <climits>
#include <cstddef>
#include <cstdint>

typedef unsigned long word;

#define WORD_BITS (sizeof(word) * CHAR_BIT)
#define BITMAP_NWORDS(_n) (((_n) + WORD_BITS - 1) / WORD_BITS)

#endif //BITMAP_GPUSSJOIN_DEFINITIONS_HXX
