#include "gpu.hxx"
#include <iostream>
#include "classes.hxx"
#include "host_timing.h"

#include "input.hxx"
#include "handler.hxx"

int gpu(const parameters& params)
{
    HostTiming hostTimings;

    bool binaryJoin = !params.foreignInput.empty();

    Collection<unsigned int> input;
    Collection<unsigned int> foreignInput;

    // Reading input
    HostTiming::Interval* readInput = hostTimings.add("Read input collection");
    add_input(params.input, input);
    hostTimings.finish(readInput);

    // Read foreign input in case of binary join
    if (binaryJoin) {
        HostTiming::Interval* readForeignInput = hostTimings.add("Read foreign-input collection");
        add_input(params.foreignInput, foreignInput);
        hostTimings.finish(readForeignInput);
    }

    auto handler = new Handler<unsigned int>(params.threshold, params.blockSize, binaryJoin);

    handler->transferInputCollection(input);

    if (binaryJoin) {
        handler->transferForeignInputCollection(foreignInput);
    }

    handler->join();

    handler->freeInputCollection();

    if (binaryJoin) {
        handler->freeForeignInputCollection();
    }

    std::cout << hostTimings << std::endl;
    std::cout << handler->getDeviceTimings() << std::endl;
    return 0;
}
