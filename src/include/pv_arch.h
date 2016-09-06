#ifndef PV_ARCH_H
#define PV_ARCH_H

#include <cMakeHeader.h> /* Loads preprocessor directives set by CMake */

/* Note: defining or undefining
 * PV_USE_OPENMP_THREADS, PV_OPENCL, PV_CUDA, PV_CUDNN
 * were moved into cMakeHeader.h on Mar 25, 2015.
 * PV_USE_MPI was moved into cMakeHeader.h on Sept 1, 2015.
 */

/* define this for 64 bit architectures */
#define PV_ARCH_64

/* define this if using OpenGL library for online graphics */
#undef PV_USE_OPENGL

/* maximum length of a path */
#define PV_PATH_MAX 256 // 127  // imageNet uses long folder names

/* Define to enable behavior convenient for parallel debugging */
#define PVP_DEBUG

/* define if using pthreads */
#undef PV_USE_PTHREADS

/* define this for the IBM Cell architecture */
#undef IBM_CELL_BE

/* define if to build for multiple threads */
#undef  MULTITHREADED
/* the maximum number of threads */
#define MAX_THREADS 1

/* controls usage of the C99 restrict keyword */
#ifndef RESTRICT
#  define RESTRICT
#endif

#endif /* PV_ARCH_H */
