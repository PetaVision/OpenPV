#ifndef PV_ARCH_H
#define PV_ARCH_H

#include "cMakeHeader.h" /* Loads preprocessor directives set by CMake */

/* Note: defining or undefining
 * PV_USE_OPENMP_THREADS, PV_OPENCL, PV_CUDA, PV_CUDNN
 * were moved into cMakeHeader.h on Mar 25, 2015.
 */

/* define this for 64 bit architectures */
#define PV_ARCH_64

/* define this if using a vendor supplied MPI library */
#define PV_USE_MPI

/* define this if using OpenGL library for online graphics */
#undef PV_USE_OPENGL

/* define this if using GDAL library to read/write images */
#define PV_USE_GDAL

#ifdef OBSOLETE // Marked obsolete Dec 9, 2014.
/* define if using SHMGET to create shared memory pool on multicore machines */
#undef USE_SHMGET
#define PAGE_SIZE 4096 // obtain by calling shell utility pagesize
#endif // OBSOLETE

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
