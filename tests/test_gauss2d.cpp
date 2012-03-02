/**
 * This file tests weight initialization to a 2D Gaussian with sigma = 1.0 and normalized to 1.0
 * Test compares HyPerConn to KernelConn,
 * assumes kernelConn produces correct 2D Gaussian weights
 *
 */

#undef DEBUG_PRINT

#include "../src/layers/HyPerLayer.hpp"
#include "../src/connections/HyPerConn.hpp"
#include "../src/connections/KernelConn.hpp"
#include "../src/io/io.h"
#include <assert.h>

#include "Example.hpp"

using namespace PV;

int check_kernel_vs_hyper(HyPerConn * cHyPer, KernelConn * cKernel, int kPre,
		int axonID);

int main(int argc, char * argv[])
{
   char * cl_args[3];
   cl_args[0] = strdup(argv[0]);
   cl_args[1] = strdup("-p");
   cl_args[2] = strdup("input/test_gauss2d.params");
   PV::HyPerCol * hc = new PV::HyPerCol("test_gauss2d column", 3, cl_args);
   PV::Example * pre = new PV::Example("test_gauss2d pre", hc);
   PV::Example * post = new PV::Example("test_gauss2d post", hc);
/*
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, InitWeights *weightInit);
 */
   InitWeights * weightInitializer = new InitWeights();
   PV::HyPerConn * cHyPer =
         new HyPerConn("test_gauss2d hyperconn", hc, pre, post,
                       CHANNEL_EXC, weightInitializer);
   PV::KernelConn * cKernel =
         new KernelConn("test_gauss2d kernelconn", hc, pre, post,
                        CHANNEL_EXC, NULL, weightInitializer);
   PV::Example * pre2 = new PV::Example("test_gauss2d pre 2", hc);
   PV::Example * post2 = new PV::Example("test_gauss2d post 2", hc);
   PV::HyPerConn * cHyPer1to2 =
         new HyPerConn("test_gauss2d hyperconn 1 to 2", hc, pre, post2,
                       CHANNEL_EXC, weightInitializer);
   PV::KernelConn * cKernel1to2 =
         new KernelConn("test_gauss2d kernelconn 1 to 2", hc, pre, post2,
                        CHANNEL_EXC, NULL, weightInitializer);
   PV::HyPerConn * cHyPer2to1 =
         new HyPerConn("test_gauss2d hyperconn 2 to 1", hc, pre2, post,
                       CHANNEL_EXC, weightInitializer);
   PV::KernelConn * cKernel2to1 =
         new KernelConn("test_gauss2d kernelconn 2 to 1", hc, pre2, post,
                        CHANNEL_EXC, NULL, weightInitializer);

   for( int c=0; c<hc->numberOfConnections(); c++ ) {
      hc->getConnection(c)->writeWeights(0, true);
   }

   const int axonID = 0;
   int num_pre_extended = pre->clayer->numExtended;
   assert(num_pre_extended == cHyPer->getNumWeightPatches());

   int status = 0;
   for (int kPre = 0; kPre < num_pre_extended; kPre++) {
     //printf("testing testing 1 2 3...\n");
     status = check_kernel_vs_hyper(cHyPer, cKernel, kPre, axonID);
     assert(status==0);
     status = check_kernel_vs_hyper(cHyPer1to2, cKernel1to2, kPre, axonID);
     assert(status==0);
     status = check_kernel_vs_hyper(cHyPer2to1, cKernel2to1, kPre, axonID);
     assert(status==0);
   }

   delete hc;
   return 0;
}

int check_kernel_vs_hyper(HyPerConn * cHyPer, KernelConn * cKernel, int kPre, int axonID)
{
   int status = 0;
   PVPatch * hyperPatch = cHyPer->getWeights(kPre, axonID);
   PVPatch * kernelPatch = cKernel->getWeights(kPre, axonID);
   int hyPerDataIndex = cHyPer->patchIndexToDataIndex(kPre);
   int kernelDataIndex = cKernel->patchIndexToDataIndex(kPre);

   int nk = cHyPer->fPatchSize() * (int) hyperPatch->nx; // hyperPatch->nf * hyperPatch->nx;
   assert(nk == (cKernel->fPatchSize() * (int) kernelPatch->nx));// assert(nk == (kernelPatch->nf * kernelPatch->nx));
   int ny = hyperPatch->ny;
   assert(ny == kernelPatch->ny);
   int sy = cHyPer->yPatchStride(); // hyperPatch->sy;
   assert(sy == cKernel->yPatchStride()); // assert(sy == kernelPatch->sy);
   pvdata_t * hyperWeights = cHyPer->get_wData(axonID, hyPerDataIndex); // hyperPatch->data;
   pvdata_t * kernelWeights = cKernel->get_wDataHead(axonID, kernelDataIndex)+hyperPatch->offset; // kernelPatch->data;
   float test_cond = 0.0f;
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         test_cond = kernelWeights[k] - hyperWeights[k];
         if (fabs(test_cond) > 0.001f) {
            printf("y %d\n", y);
            printf("k %d\n", k);
            printf("kernelweight %f\n", kernelWeights[k]);
            printf("hyperWeights %f\n", hyperWeights[k]);
            const char * cHyper_filename = "gauss2d_hyper.txt";
            cHyPer->writeTextWeights(cHyper_filename, kPre);
            const char * cKernel_filename = "gauss2d_kernel.txt";
            cKernel->writeTextWeights(cKernel_filename, kPre);
            status=1;
            //exit(EXIT_FAILURE);
         }
      }
      // advance pointers in y
      hyperWeights += sy;
      kernelWeights += sy;
   }
   return status;
}

