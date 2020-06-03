#include "host_timing.h"
#include <iostream>
#include <iomanip>

HostTiming::Interval* HostTiming::add(const std::string & descriptor)
{
    auto* interval = new HostTiming::Interval(descriptor);
    this->intervals.push_back(interval);
    interval->begin = std::chrono::steady_clock::now();
    return interval;
}

void HostTiming::finish(HostTiming::Interval* interval)
{
    interval->end = std::chrono::steady_clock::now();
}

std::ostream & operator<<(std::ostream& os, const HostTiming& timing)
{
    auto it = timing.intervals.begin();
    for(; it != timing.intervals.end(); ++it) {
        double seconds = std::chrono::duration_cast<std::chrono::microseconds>
                ((*it)->end - (*it)->begin).count() / 1000000.0;
        os << (*it)->descriptor << std::setw(11) << seconds << " secs" << std::endl;
    }
    return os;
}


HostTiming::~HostTiming()
{
    auto it = intervals.begin();
    for(; it != intervals.end(); ++it) delete *it;
}

