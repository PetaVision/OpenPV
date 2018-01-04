#ifndef ACCUMULATE_FUNCTION_HPP_
#define ACCUMULATE_FUNCTION_HPP_

#include "include/pv_common.h"

namespace PV {

void pvpatch_max_pooling(
      int kPreRes,
      int nk,
      float *RESTRICT v,
      float a,
      float *RESTRICT w,
      void *auxPtr,
      int sf);
void pvpatch_sum_pooling(
      int kPreRes,
      int nk,
      float *RESTRICT v,
      float a,
      float *RESTRICT w,
      void *auxPtr,
      int sf);

void pvpatch_max_pooling_from_post(
      int kPreRes,
      int nk,
      float *v,
      float *a,
      float *w,
      void *auxPtr,
      int sf);
void pvpatch_sum_pooling_from_post(
      int kPreRes,
      int nk,
      float *RESTRICT v,
      float *RESTRICT a,
      float *RESTRICT w,
      void *auxPtr,
      int sf);
} // namespace PV

#endif // ACCUMULATE_FUNCTION_HPP_
