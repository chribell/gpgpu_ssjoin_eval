#ifndef BITMAP_GPUSSJOIN_HOST_TIMING_HXX
#define BITMAP_GPUSSJOIN_HOST_TIMING_HXX

#include <vector>
#include <string>
#include <chrono>

class HostTiming {
public:
    struct Interval {
        typedef std::chrono::steady_clock::time_point interval_point;
        interval_point begin;
        interval_point end;
        std::string descriptor;
        explicit Interval(std::string descriptor) : descriptor(std::move(descriptor)) {}
    };
protected:
    std::vector<Interval*> intervals;

public:
    Interval* add(const std::string & descriptor);
    void finish(Interval * interval);
    ~HostTiming();
    friend std::ostream & operator<<(std::ostream & os, const HostTiming & timing);
};

#endif //BITMAP_GPUSSJOIN_HOST_TIMING_HXX
