#ifndef BITMAP_GPUSSJOIN_GPU_HXX
#define BITMAP_GPUSSJOIN_GPU_HXX

#include <string>

typedef struct {
    double threshold;
    std::string input;
    std::string foreignInput;
    unsigned int bitmap;
} parameters;

int gpu(const parameters& params);

#endif //BITMAP_GPUSSJOIN_GPU_HXX
