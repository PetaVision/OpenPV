#ifndef ACCUMULATE_FUNCTION_H_
#define ACCUMULATE_FUNCTION_H_

#ifndef CL_KERNEL_INCLUDE
#include "../include/pv_types.h"
#include "../include/pv_common.h"
//#include "../arch/opencl/pv_uint4.h"
#include "../utils/cl_random.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif
int pvpatch_accumulate(int nk, float * v, float a, pvwdata_t * w, void * auxPtr);
int pvpatch_accumulate2(int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, float* RESTRICT m);
int pvpatch_accumulate_stochastic(int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, void * auxPtr);
#ifdef OBSOLETE // Marked obsolete Aug 19, 2013.  Nobody calls pvpatch_max and whatever WTACompressedLayer was, it's not in the code now.
int pvpatch_max(int nk, float * RESTRICT v, float a, float * RESTRICT w, int feature, int * RESTRICT maxloc);
#endif // OBSOLETE
int pvpatch_max_pooling(int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, void * auxPtr);

int pvpatch_accumulate_from_post(int nk, float *v, float *a, pvwdata_t *w, float dt_factor, void * auxPtr);
int pvpatch_accumulate_stochastic_from_post(int nk, float *v, float *a, pvwdata_t *w, float dt_factor, void * auxPtr);
int pvpatch_max_pooling_from_post(int nk, float *v, float *a, pvwdata_t *w, float dt_factor, void * auxPtr);
#ifdef __cplusplus
}
#endif

#endif
