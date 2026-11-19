#ifndef COSTFUNCTIONSUM_HPP_
#define COSTFUNCTIONSUM_HPP_

#include "structures/Buffer.hpp"
#include "structures/PVLayerCube.hpp"
#include "structures/PVLayerLoc.hpp"
#include "structures/SparseList.hpp"
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
         int batchElement, PVLayerCube const *targetCube, PVLayerCube const *maskCube) const;

  private:
   float const *
   calculateBatchElementStart(float const *buffer, PVLayerLoc const *loc, int batchIndex) const;
   double calculateSumNoMask(int batchElement, PVLayerCube const *targetCube) const;
   double calculateSumWithMask(
         int batchElement, PVLayerCube const *targetCube, PVLayerCube const *maskCube) const;
   double calculateSumWithSingleFeatureMask(
         int batchElement, PVLayerCube const *targetCube, PVLayerCube const *maskCube) const;

  private:
   std::shared_ptr<C const> mCostFunction = nullptr;
};

template <class C>
float const *CostFunctionSum<C>::calculateBatchElementStart(
      float const *buffer,
      PVLayerLoc const *loc,
      int batchIndex) const {
   int nxExt        = loc->nx + loc->halo.lt + loc->halo.rt;
   int nyExt        = loc->ny + loc->halo.dn + loc->halo.up;
   long numExtended = (long)nxExt * (long)nyExt * (long)loc->nf;
   return &buffer[batchIndex * numExtended];
}

template <class C>
double CostFunctionSum<C>::calculateSum(
         int batchElement, PVLayerCube const *targetCube, PVLayerCube const *maskCube) const {
   pvAssert(targetCube);
   double result;
   if (maskCube) {
      pvAssert(maskCube->loc.nx == targetCube->loc.nx);
      pvAssert(maskCube->loc.ny == targetCube->loc.ny);
      pvAssert(maskCube->loc.nf == targetCube->loc.nf or maskCube->loc.nf == 1);
      if (targetCube->loc.nf > 1 and maskCube->loc.nf == 1) {
         result = calculateSumWithSingleFeatureMask(batchElement, targetCube, maskCube);
      }
      else {
         result = calculateSumWithMask(batchElement, targetCube, maskCube);
      }
   }
   else {
      result = calculateSumNoMask(batchElement, targetCube);
   }
   return result;
}

template <class C>
double
CostFunctionSum<C>::calculateSumNoMask(int batchElement, PVLayerCube const *targetCube) const {
   int const nx = targetCube->loc.nx;
   int const ny = targetCube->loc.ny;
   int const nf = targetCube->loc.nf;
   int const lt = targetCube->loc.halo.lt;
   int const rt = targetCube->loc.halo.rt;
   int const dn = targetCube->loc.halo.dn;
   int const up = targetCube->loc.halo.up;

   C const &costFunction = *(mCostFunction.get());
   double sum            = 0.0;
   if (targetCube->isSparse) {
      SparseList<float> const *targetSparseList = targetCube->activeIndices[batchElement];
      auto const &targetSparseData              = targetSparseList->getContents();
      for (auto const &e : targetSparseData) {
         double v = static_cast<double>(e.value);
         sum += costFunction(v);
      }
   }
   else {
      float const *targetData = targetCube->data[batchElement];
      long const numNeurons = static_cast<long>(nx) * static_cast<long>(ny) * static_cast<long>(nf);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
      for (long k = 0; k < numNeurons; ++k) {
         long kex  = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         double v = static_cast<double>(targetData[kex]);
         sum += costFunction(v);
      }
   }
   return sum;
}

template <class C>
double CostFunctionSum<C>::calculateSumWithMask(
         int batchElement, PVLayerCube const *targetCube, PVLayerCube const *maskCube) const {
   int const nx = targetCube->loc.nx;
   int const ny = targetCube->loc.ny;
   int const nf = targetCube->loc.nf;
   int const lt = targetCube->loc.halo.lt;
   int const rt = targetCube->loc.halo.rt;
   int const dn = targetCube->loc.halo.dn;
   int const up = targetCube->loc.halo.up;

   int const mask_lt = maskCube->loc.halo.lt;
   int const mask_rt = maskCube->loc.halo.rt;
   int const mask_dn = maskCube->loc.halo.dn;
   int const mask_up = maskCube->loc.halo.up;

   C const &costFunction   = *(mCostFunction.get());
   double sum              = 0.0;
   float const *targetData = nullptr;
   float const *maskData   = nullptr;
   long const numNeurons   = static_cast<long>(nx) * static_cast<long>(ny) * static_cast<long>(nf);
   Buffer<float> targetDenseBuffer, maskDenseBuffer;
   if (targetCube->isSparse) {
      targetDenseBuffer.resize(nx + lt + rt, ny + dn + up, nf);
      SparseList<float> const *sparseList = targetCube->activeIndices[batchElement];
      sparseList->toBuffer(targetDenseBuffer, 0.0f);
      targetData = targetDenseBuffer.asVector().data();
   }
   else {
      targetData = targetCube->data[batchElement];
   }
   if (maskCube->isSparse) {
      maskDenseBuffer.resize(nx + lt + rt, ny + dn + up, nf);
      SparseList<float> const *sparseList = maskCube->activeIndices[batchElement];
      sparseList->toBuffer(maskDenseBuffer, 0.0f);
      maskData = maskDenseBuffer.asVector().data();
   }
   else {
      maskData = maskCube->data[batchElement];
   }
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
   for (long k = 0; k < numNeurons; ++k) {
      long kexMask = kIndexExtended(k, nx, ny, nf, mask_lt, mask_rt, mask_dn, mask_up);
      if (maskData[kexMask] != 0.0f) {
         long kex  = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         double v = static_cast<double>(targetData[kex]);
         sum += costFunction(v);
      }
   }
   return sum;
}

template <class C>
double CostFunctionSum<C>::calculateSumWithSingleFeatureMask(
         int batchElement, PVLayerCube const *targetCube, PVLayerCube const *maskCube) const {
   int const nx = targetCube->loc.nx;
   int const ny = targetCube->loc.ny;
   int const nf = targetCube->loc.nf;
   int const lt = targetCube->loc.halo.lt;
   int const rt = targetCube->loc.halo.rt;
   int const dn = targetCube->loc.halo.dn;
   int const up = targetCube->loc.halo.up;

   int const mask_lt = maskCube->loc.halo.lt;
   int const mask_rt = maskCube->loc.halo.rt;
   int const mask_dn = maskCube->loc.halo.dn;
   int const mask_up = maskCube->loc.halo.up;

   C const &costFunction = *(mCostFunction.get());
   double sum              = 0.0;
   float const *targetData = nullptr;
   float const *maskData   = nullptr;
   long const nxny         = static_cast<long>(nx) * static_cast<long>(ny);
   Buffer<float> targetDenseBuffer, maskDenseBuffer;
   if (targetCube->isSparse) {
      targetDenseBuffer.resize(nx + lt + rt, ny + dn + up, nf);
      SparseList<float> const *sparseList = targetCube->activeIndices[batchElement];
      sparseList->toBuffer(targetDenseBuffer, 0.0f);
      targetData = targetDenseBuffer.asVector().data();
   }
   else {
      targetData = targetCube->data[batchElement];
   }
   if (maskCube->isSparse) {
      maskDenseBuffer.resize(nx + lt + rt, ny + dn + up, 1);
      SparseList<float> const *sparseList = maskCube->activeIndices[batchElement];
      sparseList->toBuffer(maskDenseBuffer, 0.0f);
      maskData = maskDenseBuffer.asVector().data();
   }
   else {
      maskData = maskCube->data[batchElement];
   }
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
   for (long k = 0; k < nxny; ++k) {
      long kexMask = kIndexExtended(k, nx, ny, 1, mask_lt, mask_rt, mask_dn, mask_up);
      if (maskData[kexMask]) {
         long const xyBase = k * nf;
         for (int f = 0; f < nf; ++f) {
            long kex  = kIndexExtended(xyBase + f, nx, ny, nf, lt, rt, dn, up);
            double v = (double)targetData[kex];
            sum += costFunction(v);
         }
      }
   }
   return sum;
}

} // namespace PV

#endif // COSTFUNCTIONSUM_HPP_
