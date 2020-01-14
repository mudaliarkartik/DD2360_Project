#ifndef TIMING_H
#define TIMING_H

#ifndef _WIN64
#include <sys/time.h>
#endif

// return time in second
double cpuSecond();

#endif
