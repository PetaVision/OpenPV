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

/* define if using SHMGET to create shared memory pool on multicore machines */
#undef USE_SHMGET
#define PAGE_SIZE 4096 // obtain by calling shell utility pagesize

/* maximum length of a path */
#define PV_PATH_MAX 256 // 127  // imageNet uses long folder names

/* Define to enable behavior convenient for parallel debugging */
#undef PVP_DEBUG

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
