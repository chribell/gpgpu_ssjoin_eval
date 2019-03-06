#include "gpu.hxx"
#include <string>
#include <iostream>
#include <fstream>
#include "classes.hxx"
#include "host_timing.h"

#include "input.hxx"
#include "handler.hxx"

int gpu(parameters params)
{
    Collection<unsigned int> collection;
    HostTiming hostTimings;

    // Reading input
    HostTiming::Interval* readInput = hostTimings.add("Read input collection");
    add_input(params.input, collection);
    hostTimings.finish(readInput);

    auto handler = new Handler<unsigned int>(params.threshold);

    handler->transferCollection(collection);
    handler->constructBitmaps(BITMAP_NWORDS(params.bitmap));

    handler->join();

    handler->freeCollection();

    std::cout << hostTimings << std::endl;
    std::cout << handler->getDeviceTimings() << std::endl;
    return 0;
}
