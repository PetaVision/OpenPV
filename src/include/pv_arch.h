#ifndef PV_ARCH_H
#define PV_ARCH_H

// Preprocessor directives set by CMake
#include <cMakeHeader.h>

// Maximum length of a path
#define PV_PATH_MAX 256

// Controls usage of the C99 restrict keyword
#ifndef RESTRICT
#define RESTRICT
#endif

#endif
