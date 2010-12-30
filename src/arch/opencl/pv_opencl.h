/*
 * pv_opencl.h
 *
 *  Created on: Jul 30, 2010
 *      Author: Craig Rasmussen
 */

#ifndef PV_OPENCL_H_
#define PV_OPENCL_H_

typedef struct uint4_ {
   unsigned int s0, s1, s2, s3;
} uint4;

#ifdef PV_USE_OPENCL
#  include <OpenCL/opencl.h>

#ifdef PV_USE_TAU
#  include <TAU.h>
#  include <Profile/TauGpuAdapterOpenCLExp.h>
#endif

// OpenCL attributes
//
#  define CL_KERNEL       __kernel
#  define CL_MEM_GLOBAL   __global
#  define CL_MEM_LOCAL    __local

#else

#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_LOCAL

#  define cl_uint           unsigned int

#  define cl_mem_flags      int
#  define cl_device_id      int
#  define cl_context        int
#  define cl_command_queue  int
#  define cl_kernel         int
#  define cl_program        int
#  define cl_event          int

#  define CL_SUCCESS            0
#  define CL_MEM_READ_ONLY      0
#  define CL_MEM_WRITE_ONLY     0
#  define CL_MEM_USE_HOST_PTR   0
#  define CL_MEM_COPY_HOST_PTR  0

#endif

////////////////////////////////////////////////////////////////////////////////

// guess at maxinum number of devices (CPU + GPU)
//
#define MAX_DEVICES (2)

// guess at maximum work item dimensions
//
#define MAX_WORK_ITEM_DIMENSIONS (3)

#define PVCL_GET_DEVICE_ID_FAILURE    1
#define PVCL_CREATE_CONTEXT_FAILURE   2
#define PVCL_CREATE_CMD_QUEUE_FAILURE 3
#define PVCL_CREATE_PROGRAM_FAILURE   4
#define PVCL_BUILD_PROGRAM_FAILURE    5
#define PVCL_CREATE_KERNEL_FAILURE    6

#endif /* PV_OPENCL_H_ */
