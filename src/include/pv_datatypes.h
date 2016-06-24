/*
 *
 * Created Sep 12, 2014
 * Author: peteschultz
 */

// Defines datatypes for connection weights, activity buffers, membrane potentials, and GSyn buffers.
//
// Currently, only float is supported for all these types.
//
// This file is included by OpenCL kernels as well as C and C++ source files, so should be kept to minimal preprocessor directives.

#ifndef PV_DATATYPES_H_
#define PV_DATATYPES_H_

// The data type for weights
#define pvwdata_t              float
#define PV_WCAST               float

// The data type for activity and datastore data
#define pvadata_t              float
#define PV_ACAST               float
#define max_pvadata_t FLT_MAX
#define min_pvadata_t FLT_MIN

// The data type for a layer's V buffer
#define pvpotentialdata_t      float
#define max_pvvdata_t FLT_MAX
#define min_pvvdata_t FLT_MIN

// The data type for a layer's GSyn buffers
#define pvgsyndata_t           float
 
// Catch-all datatype, deprecated
#define pvdata_t               float

// Note: conductances have a datatype pvconductance_t, but since conductances are
// specific to the LIF class, its derived classes, and LIF-specific probes,
// pvconductance_t is defined in <kernels/LIFparams.h>
 
#endif // PV_DATATYPES_H_
