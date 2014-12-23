/*
 * CliqueApplyConn.cpp
 *
 *  Created on: Sep 8, 2011
 *      Author: gkenyon
 */

#include "CliqueApplyConn.hpp"

namespace PV {

CliqueApplyConn::CliqueApplyConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            ChannelType channel, const char * filename, InitWeights *weightInit) {

   CliqueApplyConn::initialize(name, hc, pre, post, channel, filename, weightInit);
}

int CliqueApplyConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename,
      InitWeights *weightInit)
{
   return KernelConn::initialize(name, hc, pre, post, channel, filename, weightInit);
}


int sumCliqueWeights(PVPatch * targetWpatch, PVPatch * distractorWpatch, double * sum_denom, double * sum_weights, HyPerConn * parentConn)
{
   assert(targetWpatch != NULL);
   pvdata_t * targetWeights = targetWpatch->data;
   assert(targetWeights != NULL);
   const int nx = targetWpatch->nx;
   const int ny = targetWpatch->ny;
   const int nf = parentConn->fPatchSize();
   const int sy = parentConn->yPatchStride();
   assert(distractorWpatch != NULL);
   pvdata_t * distractorWeights = distractorWpatch->data;
   assert(distractorWeights != NULL);
   assert(nx == distractorWpatch->nx);
   assert(ny == distractorWpatch->ny);
   //assert(nf == distractorWpatch->nf);
   double sum_denom_tmp = 0;
   double sum_weights_tmp = 0;
   for (int ky = 0; ky < ny; ky++) {
      for(int iWeight = 0; iWeight < nf * nx; iWeight++ ){
         pvdata_t denom_tmp = targetWeights[iWeight] + distractorWeights[iWeight];
         sum_denom_tmp += (denom_tmp != 0.0f) ? (1.0f / denom_tmp) : 0.0f;
         pvdata_t numerator_temp = targetWeights[iWeight] - distractorWeights[iWeight];
         sum_weights_tmp += (denom_tmp != 0.0f) ? (numerator_temp / denom_tmp) : 0.0f;
      }
      targetWeights += sy;
      distractorWeights += sy;
   }
   *sum_denom = sum_denom_tmp;
   *sum_weights = sum_weights_tmp;
   return PV_SUCCESS;
} // sumCliqueWeights


int scaleCliqueWeights(PVPatch * targetWpatch, PVPatch * distractorWpatch, double sum_denom, double sum_weights, HyPerConn * parentConn)
{
   assert(targetWpatch != NULL);
   pvdata_t * targetWeights = targetWpatch->data;
   assert(targetWeights != NULL);
   const int nx = targetWpatch->nx;
   const int ny = targetWpatch->ny;
   const int nf = parentConn->fPatchSize();
   const int sy = parentConn->yPatchStride();
   assert(distractorWpatch != NULL);
   pvdata_t * distractorWeights = distractorWpatch->data;
   assert(distractorWeights != NULL);
   assert(nx == distractorWpatch->nx);
   assert(ny == distractorWpatch->ny);
   //assert(nf == distractorWpatch->nf);
   pvdata_t shift_val = 0; //(sum_denom != 0.0f) ? ((1.0f/2.0f) * sum_weights / sum_denom) : 0.0f;
   for (int ky = 0; ky < ny; ky++) {
      for(int iWeight = 0; iWeight < nf * nx; iWeight++ ){
         pvdata_t denom_tmp = targetWeights[iWeight] + distractorWeights[iWeight];
         targetWeights[iWeight] = (denom_tmp != 0.0f) ? ((targetWeights[iWeight] - shift_val) / denom_tmp) : 0.0f;
         distractorWeights[iWeight] = (denom_tmp != 0.0f) ? ((distractorWeights[iWeight] + shift_val) / denom_tmp) : 0.0f;
      }
      targetWeights += sy;
      distractorWeights += sy;
  }
   return PV_SUCCESS;
} // scaleWeights


int CliqueApplyConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId)
{
   int status = PV_SUCCESS;
   const int num_kernels = numDataPatches();

   // individually normalize each arbor
   assert(this->normalize_arbors_individually);
   status = NoSelfKernelConn::normalizeWeights(patches, numPatches, arborId);  // parent class should return PV_BREAK
   assert( (status == PV_SUCCESS) || (status == PV_BREAK) );

   //  now normalize both arbors together to force sumAll == 0
   //assert(this->numberOfAxonalArborLists() == 2);
   if (this->numberOfAxonalArborLists() != 2) { // aroborId == 0 -> target kernel, aroborId == 1 -> distractor kernel
      fprintf(stderr, "CliqueApplyConn::%s::normalizeWeights, numAxonalArbors = %d != 2", this->name, this->numberOfAxonalArborLists());
      fprintf(stderr, "\n");
      return PV_FAILURE;
   }
   double sum_weights = 0.0f;
   double sum_denom = 0.0f;
   float tol = 0.0f; //0.001;
   int loop_count = 0;
   int max_loop_count = 1;
   for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
      PVPatch * targetWpatch = this->getKernelPatch(0,kPatch);
      PVPatch * distractorWpatch = this->getKernelPatch(1,kPatch);
      status = sumCliqueWeights(targetWpatch, distractorWpatch, &sum_denom, &sum_weights, this);
      assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
      loop_count = 0;
      while((fabs(sum_denom) > tol) && (loop_count < max_loop_count)){
         status = scaleCliqueWeights(targetWpatch, distractorWpatch, sum_denom, sum_weights, this);
         assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
         status = sumCliqueWeights(targetWpatch, distractorWpatch, &sum_denom, &sum_weights, this);
         assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
         loop_count++;
       } // while
      if (loop_count > max_loop_count) { // aroborId == 0 -> target kernel, aroborId == 1 -> distractor kernel
         fprintf(stderr, "CliqueApplyConn::%s::normalizeWeights, loop_count = %d >= max_loop_count = %d", this->name, loop_count, max_loop_count);
         fprintf(stderr, "\n");
         return PV_FAILURE;
      }
      // change sign of distractor weights
      pvdata_t * distractorWeights = distractorWpatch->data;
      assert(distractorWeights != NULL);
      const int nx = targetWpatch->nx;
      const int ny = targetWpatch->ny;
      const int nf = nfp;
      const int sy = syp;
      for (int ky = 0; ky < ny; ky++) {
         for(int iWeight = 0; iWeight < nf * nx; iWeight++ ){
            distractorWeights[iWeight] *= -1.0f;
         }
         distractorWeights += sy;
     }
   } // kPatch < numPatches

   return PV_BREAK;
} // normalizeWeights

} // namespace PV

