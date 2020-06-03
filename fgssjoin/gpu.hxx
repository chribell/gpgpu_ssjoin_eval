#ifndef FGSSJOIN_GPU_HXX
#define FGSSJOIN_GPU_HXX

#include <string>

typedef struct {
    double threshold;
    std::string input;
    std::string foreignInput;
    unsigned int blockSize;
} parameters;

int gpu(const parameters& params);

#endif //FGSSJOIN_GPU_HXX
