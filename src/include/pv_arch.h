#ifndef PV_ARCH_H
#define PV_ARCH_H

/* define this for 64 bit architectures */
#define PV_ARCH_64

/* define this if using a vendor supplied MPI library */
#define PV_USE_MPI

/* define this if using OpenCL for threads on CPU or GPU */
#undef PV_USE_OPENCL

/* define this if using OpenGL library for online graphics */
#undef PV_USE_OPENGL

/* define this if using GDAL library to read/write images */
#define PV_USE_GDAL

/* maximum length of a path */
#define PV_PATH_MAX 256 // 127  // imageNet uses long folder names

/* define if using pthreads */
#undef PV_USE_PTHREADS

/* define this for the IBM Cell architecture */
#undef IBM_CELL_BE

/* define if to build for multiple threads */
#undef  MULTITHREADED
/* the maximum number of threads */
#define MAX_THREADS 1

/* For building using Eclipse */
#undef ECLIPSE

/* controls usage of the C99 restrict keyword */
#ifndef RESTRICT
#  define RESTRICT
#endif

#endif /* PV_ARCH_H */
