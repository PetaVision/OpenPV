#ifndef ACCUMULATE_FUNCTIONS_HPP_
#define ACCUMULATE_FUNCTIONS_HPP_

#include "include/pv_common.h"

namespace PV {

void pvpatch_max_pooling(
      int kPreRes,
      int nk,
      float *RESTRICT v,
      float a,
      float const *RESTRICT w,
      void *auxPtr,
      int sf);
void pvpatch_sum_pooling(
      int kPreRes,
      int nk,
      float *RESTRICT v,
      float a,
      float const *RESTRICT w,
      void *auxPtr,
      int sf);

void pvpatch_max_pooling_from_post(
      int kPreRes,
      int nk,
      float *v,
      float const *a,
      float const *w,
      void *auxPtr,
      int sf);
void pvpatch_sum_pooling_from_post(
      int kPreRes,
      int nk,
      float *RESTRICT v,
      float const *RESTRICT a,
      float const *RESTRICT w,
      void *auxPtr,
      int sf);
} // namespace PV

#endif // ACCUMULATE_FUNCTIONS_HPP_
