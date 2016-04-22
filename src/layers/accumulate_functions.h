#ifndef ACCUMULATE_FUNCTION_H_
#define ACCUMULATE_FUNCTION_H_

#ifndef CL_KERNEL_INCLUDE
#include "../include/pv_types.h"
#include "../include/pv_common.h"
#include "../utils/cl_random.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif
  int pvpatch_accumulate(int kPreRes, int nk, float * v, float a, pvwdata_t * w, void * auxPtr, int sf);
  int pvpatch_accumulate2(int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, float* RESTRICT m);
  int pvpatch_accumulate_stochastic(int kPreRes, int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, void * auxPtr, int sf);
  int pvpatch_max_pooling(int kPreRes, int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, void * auxPtr, int sf);
  int pvpatch_sum_pooling(int kPreRes, int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, void * auxPtr, int sf);

  int pvpatch_accumulate_from_post(int kPreRes, int nk, float *v, float *a, pvwdata_t *w, float dt_factor, void * auxPtr, int sf);
  int pvpatch_accumulate_stochastic_from_post(int kPreRes, int nk, float *v, float *a, pvwdata_t *w, float dt_factor, void * auxPtr, int sf);
  int pvpatch_max_pooling_from_post(int kPreRes, int nk, float *v, float *a, pvwdata_t *w, float dt_factor, void * auxPtr, int sf);
  int pvpatch_sumpooling_from_post(int kPreRes, int nk, float * RESTRICT v, float * RESTRICT a, pvwdata_t * RESTRICT w, float dt_factor, void * auxPtr, int sf);
#ifdef __cplusplus
}
#endif

#endif
