#include "Timing.h"

#ifdef _WIN64
#include "timewindows.h"
#endif

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
