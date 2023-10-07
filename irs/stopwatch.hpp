#pragma once

#include <chrono>


/******************************************************************************
template class for starting/stopping a stopwatch and returning the elapsed time
(in seconds) when stopped
******************************************************************************/
class Stopwatch
{
    std::chrono::high_resolution_clock::time_point t_start;
    std::chrono::high_resolution_clock::time_point t_end;

public:

    void start()
    {
        t_start = std::chrono::high_resolution_clock::now();
    }

    double stop(bool reset = true)
    {
        t_end = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0;

        if (reset)
        {
            t_start = std::chrono::high_resolution_clock::time_point();
            t_end = std::chrono::high_resolution_clock::time_point();
        }

        return dt;
    }

};

