#pragma once

#include <chrono> 


/******************************************************************************
template class for starting/stopping a stopwatch and returning the elapsed time
(in seconds) when stopped
******************************************************************************/
class Stopwatch
{
    std::chrono::high_resolution_clock::time_point tstart;
    std::chrono::high_resolution_clock::time_point tend;

public:

    void start()
    {
        tstart = std::chrono::high_resolution_clock::now();
    }

    double stop(bool reset = true)
    {
        tend = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() / 1000.0;

        if (reset)
        {
            tstart = std::chrono::high_resolution_clock::time_point();
            tend = std::chrono::high_resolution_clock::time_point();
        }

        return dt;
    }

};

