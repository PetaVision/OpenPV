#ifndef PV_ARCH_H
#define PV_ARCH_H

/* define this for 64 bit architectures */
#undef PV_ARCH_64

/* maximum length of a path */
#define PV_PATH_MAX 127

/* define this for the IBM Cell architecture */
#undef IBM_CELL_BE

/* define if to build for multiple threads */
#undef  MULTITHREADED
/* the maximum number of threads */
#define MAX_THREADS 1

/* For building using Eclipse */
#define ECLIPSE

/* controls usage of the C99 restrict keyword */
#ifndef RESTRICT
#  define RESTRICT
#endif

#endif /* PV_ARCH_H */
