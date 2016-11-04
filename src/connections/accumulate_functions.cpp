#include <connections/accumulate_functions.hpp>

// TODO athresher 9-12-16: Why do these have different return types?

#ifdef COMPRESS_PHI
void pvpatch_accumulate(
      int kPreExt,
      int nk,
      float *restrict v,
      float a,
      float *restrict w,
      float *restrict m) {
   const float scale     = 33.3f;
   const float inv_scale = 1.0f / scale;
   const float shift     = 2.0f;
   for (int k = 0; k < nk; ++k) {
      v[k] = (shift + scale * v[k] + a * w[k] * m[k] - shift) * inv_scale;
   }
}
#else
int pvpatch_accumulate(
      int kPreExt,
      int nk,
      float *RESTRICT v,
      float a,
      float *RESTRICT w,
      void *auxPtr,
      int sf) {
   for (int k = 0; k < nk; ++k) {
      v[k] += a * w[k];
   }
   return PV_SUCCESS;
}
#endif

int pvpatch_accumulate_from_post(
      int kPreExt,
      int nk,
      float *RESTRICT v,
      float *RESTRICT a,
      float *RESTRICT w,
      float dt_factor,
      void *auxPtr,
      int sf) {
   float dv = 0.0f;
   for (int k = 0; k < nk; ++k) {
      dv += a[k] * w[k];
   }
   *v += dt_factor * dv;
   return PV_SUCCESS;
}

int pvpatch_accumulate2(int nk, float *RESTRICT v, float a, float *RESTRICT w, float *RESTRICT m) {
   for (int k = 0; k < nk; ++k) {
      v[k] += a * w[k] * m[k];
   }
   return PV_SUCCESS;
}

// TODO athresher 9-12-16: Can we change this void pointer to *taus_uint4?
int pvpatch_accumulate_stochastic(
      int kPreExt,
      int nk,
      float *RESTRICT v,
      float a,
      float *RESTRICT w,
      void *auxPtr,
      int sf) {
   taus_uint4 *rng = (taus_uint4 *)auxPtr;
   long along      = (long)((double)a * cl_random_max());
   for (int k = 0; k < nk; k += sf) {
      *rng = cl_random_get(*rng);
      v[k] += (rng->s0 < along) * w[k];
   }
   return PV_SUCCESS;
}

int pvpatch_accumulate_stochastic_from_post(
      int kPreExt,
      int nk,
      float *RESTRICT v,
      float *RESTRICT a,
      float *RESTRICT w,
      float dt_factor,
      void *auxPtr,
      int sf) {
   taus_uint4 *rng = (taus_uint4 *)auxPtr;
   float dv        = 0.0f;
   for (int k = 0; k < nk; k += sf) {
      *rng     = cl_random_get(*rng);
      double p = (double)rng->s0 / cl_random_max(); // 0.0 < p < 1.0
      dv += ((float)p < a[k] * dt_factor) * w[k];
   }
   *v = *v + dv;
   return PV_SUCCESS;
}

int pvpatch_max_pooling(
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
   return PV_SUCCESS;
}

int pvpatch_max_pooling_from_post(
      int kPreGlobalExt,
      int nk,
      float *RESTRICT v,
      float *RESTRICT a,
      float *RESTRICT w,
      float dt_factor,
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
   return PV_SUCCESS;
}

int pvpatch_sum_pooling(
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
   return PV_SUCCESS;
}

int pvpatch_sumpooling_from_post(
      int kPreExt,
      int nk,
      float *RESTRICT v,
      float *RESTRICT a,
      float *RESTRICT w,
      float dt_factor,
      void *auxPtr,
      int sf) {
   float dv = 0.0f;
   for (int k = 0; k < nk; k += sf) {
      dv += a[k];
   }
   *v += dt_factor * dv * w[0];
   return PV_SUCCESS;
}
