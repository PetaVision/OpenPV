#ifndef COSTFUNCTIONSUM_HPP_
#define COSTFUNCTIONSUM_HPP_

#include "include/PVLayerLoc.h"
#include "utils/PVAssert.hpp"
#include "utils/conversions.hpp"
#include <memory>

namespace PV {

template <class C>
class CostFunctionSum {
  public:
   CostFunctionSum(std::shared_ptr<C const> costFunction) : mCostFunction(costFunction) {}
   ~CostFunctionSum() {}
   double calculateSum(
         float const *buffer,
         PVLayerLoc const *bufferLoc,
         float const *mask,
         PVLayerLoc const *maskLoc,
         int batchIndex) const;

  private:
   float const *
   calculateBatchElementStart(float const *buffer, PVLayerLoc const *loc, int batchIndex) const;
   double calculateSumNoMask(float const *buffer, PVLayerLoc const *bufferLoc) const;
   double calculateSumWithMask(
         float const *buffer,
         PVLayerLoc const *bufferLoc,
         float const *mask,
         PVLayerLoc const *maskLoc) const;
   double calculateSumWithSingleFeatureMask(
         float const *buffer,
         PVLayerLoc const *bufferLoc,
         float const *mask,
         PVLayerLoc const *maskLoc) const;

  private:
   std::shared_ptr<C const> mCostFunction = nullptr;
};

template <class C>
float const *CostFunctionSum<C>::calculateBatchElementStart(
      float const *buffer,
      PVLayerLoc const *loc,
      int batchIndex) const {
   int nxExt       = loc->nx + loc->halo.lt + loc->halo.rt;
   int nyExt       = loc->ny + loc->halo.dn + loc->halo.up;
   int numExtended = nxExt * nyExt * loc->nf;
   return &buffer[batchIndex * numExtended];
}

template <class C>
double CostFunctionSum<C>::calculateSum(
      float const *buffer,
      PVLayerLoc const *bufferLoc,
      float const *mask,
      PVLayerLoc const *maskLoc,
      int batchIndex) const {
   pvAssert(buffer);
   pvAssert(bufferLoc);
   if (mask) {
      pvAssert(maskLoc);
      pvAssert(maskLoc->nx == bufferLoc->nx);
      pvAssert(maskLoc->ny == bufferLoc->ny);
      pvAssert(maskLoc->nf == bufferLoc->nf or maskLoc->nf == 1);
   }
   double result;
   float const *bufferElement = calculateBatchElementStart(buffer, bufferLoc, batchIndex);
   if (mask) {
      float const *maskElement = calculateBatchElementStart(mask, maskLoc, batchIndex);
      if (bufferLoc->nf > 1 and maskLoc->nf == 1) {
         result = calculateSumWithSingleFeatureMask(bufferElement, bufferLoc, maskElement, maskLoc);
      }
      else {
         result = calculateSumWithMask(bufferElement, bufferLoc, maskElement, maskLoc);
      }
   }
   else {
      result = calculateSumNoMask(bufferElement, bufferLoc);
   }
   return result;
}

template <class C>
double
CostFunctionSum<C>::calculateSumNoMask(float const *buffer, PVLayerLoc const *bufferLoc) const {
   int const nx = bufferLoc->nx;
   int const ny = bufferLoc->ny;
   int const nf = bufferLoc->nf;
   int const lt = bufferLoc->halo.lt;
   int const rt = bufferLoc->halo.rt;
   int const dn = bufferLoc->halo.dn;
   int const up = bufferLoc->halo.up;

   C const &costFunction = *(mCostFunction.get());
   double sum            = 0.0;
   int const numNeurons  = nx * ny * nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
   for (int k = 0; k < numNeurons; ++k) {
      int kex  = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      double v = (double)buffer[kex];
      sum += costFunction(v);
   }
   return sum;
}

template <class C>
double CostFunctionSum<C>::calculateSumWithMask(
      float const *buffer,
      PVLayerLoc const *bufferLoc,
      float const *mask,
      PVLayerLoc const *maskLoc) const {
   int const nx = bufferLoc->nx;
   int const ny = bufferLoc->ny;
   int const nf = bufferLoc->nf;
   int const lt = bufferLoc->halo.lt;
   int const rt = bufferLoc->halo.rt;
   int const dn = bufferLoc->halo.dn;
   int const up = bufferLoc->halo.up;

   int const mask_lt = maskLoc->halo.lt;
   int const mask_rt = maskLoc->halo.rt;
   int const mask_dn = maskLoc->halo.dn;
   int const mask_up = maskLoc->halo.up;

   C const &costFunction = *(mCostFunction.get());
   double sum            = 0.0;
   int const numNeurons  = nx * ny * nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
   for (int k = 0; k < numNeurons; ++k) {
      int kexMask = kIndexExtended(k, nx, ny, nf, mask_lt, mask_rt, mask_dn, mask_up);
      if (mask[kexMask] != 0.0f) {
         int kex  = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         double v = (double)buffer[kex];
         sum += costFunction(v);
      }
   }
   return sum;
}

template <class C>
double CostFunctionSum<C>::calculateSumWithSingleFeatureMask(
      float const *buffer,
      PVLayerLoc const *bufferLoc,
      float const *mask,
      PVLayerLoc const *maskLoc) const {
   int const nx = bufferLoc->nx;
   int const ny = bufferLoc->ny;
   int const nf = bufferLoc->nf;
   int const lt = bufferLoc->halo.lt;
   int const rt = bufferLoc->halo.rt;
   int const dn = bufferLoc->halo.dn;
   int const up = bufferLoc->halo.up;

   int const mask_lt = maskLoc->halo.lt;
   int const mask_rt = maskLoc->halo.rt;
   int const mask_dn = maskLoc->halo.dn;
   int const mask_up = maskLoc->halo.up;

   C const &costFunction = *(mCostFunction.get());
   double sum            = 0.0;
   int const nxny        = nx * ny;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
   for (int k = 0; k < nxny; ++k) {
      int kexMask = kIndexExtended(k, nx, ny, 1, mask_lt, mask_rt, mask_dn, mask_up);
      if (mask[kexMask]) {
         int const xyBase = k * nf;
         for (int f = 0; f < nf; ++f) {
            int kex  = kIndexExtended(xyBase + f, nx, ny, nf, lt, rt, dn, up);
            double v = (double)buffer[kex];
            sum += costFunction(v);
         }
      }
   }
   return sum;
}

} // namespace PV

#endif // COSTFUNCTIONSUM_HPP_
