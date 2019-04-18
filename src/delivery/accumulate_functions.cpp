#include "delivery/accumulate_functions.hpp"

namespace PV {

void pvpatch_max_pooling(
      int kPreGlobalExt,
      int nk,
      float *RESTRICT v,
      float a,
      float *RESTRICT w,
      void *auxPtr,
      int sf) {
   float *gate = (float *)auxPtr;
   for (int k = 0; k < nk; k += sf) {
      float checkVal = a * w[0];
      if (v[k] <= checkVal) {
         v[k] = checkVal;
         if (gate) {
            gate[k] = (float)kPreGlobalExt;
         }
      }
   }
}

void pvpatch_max_pooling_from_post(
      int kPreGlobalExt,
      int nk,
      float *RESTRICT v,
      float *RESTRICT a,
      float *RESTRICT w,
      void *auxPtr,
      int sf) {
   float vmax  = *v;
   float *gate = (float *)auxPtr;
   int gateMax = 0;
   if (gate) {
      gateMax = (int)*gate;
   }
   for (int k = 0; k < nk; k += sf) {
      float checkVal = a[k];
      if (vmax <= checkVal) {
         vmax    = checkVal;
         gateMax = kPreGlobalExt + k;
      }
   }
   *v = vmax * w[0];
   if (gate) {
      *gate = (float)gateMax;
   }
}

void pvpatch_sum_pooling(
      int kPreExt,
      int nk,
      float *RESTRICT v,
      float a,
      float *RESTRICT w,
      void *auxPtr,
      int sf) {
   for (int k = 0; k < nk; k += sf) {
      v[k] += a * w[0];
   }
}

void pvpatch_sum_pooling_from_post(
      int kPreExt,
      int nk,
      float *RESTRICT v,
      float *RESTRICT a,
      float *RESTRICT w,
      void *auxPtr,
      int sf) {
   float dv = 0.0f;
   for (int k = 0; k < nk; k += sf) {
      dv += a[k];
   }
   *v += dv * w[0];
}

} // namespace PV
