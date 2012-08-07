/**
 * This file tests weight initialization based on cocircularity
 * Test compares HyPerConn to CocircConn,
 * assumes CocircConn produces correct weights
 *
 */

#undef DEBUG_PRINT

#include "Example.hpp"
#include "../src/layers/HyPerLayer.hpp"
#include "../src/connections/HyPerConn.hpp"
#include "../src/connections/KernelConn.hpp"
// #include "../src/connections/CocircConn.hpp" // Cocirc now implemented in KernelConn using initWeights
#include "../src/io/io.h"
#include <assert.h>

using namespace PV;

int check_cocirc_vs_hyper(HyPerConn * cHyPer, KernelConn * cKernel, int kPre,
		int axonID);

int main(int argc, char * argv[])
{
   PV::HyPerCol * hc = new PV::HyPerCol("test_cocirc column", argc, argv);
   PV::Example * pre = new PV::Example("test_cocirc pre", hc);
   PV::Example * post = new PV::Example("test_cocirc post", hc);
   PV::HyPerConn * cHyPer = new HyPerConn("test_cocirc hyperconn", hc, pre,
                                          post);
   PV::KernelConn * cCocirc = new KernelConn("test_cocirc cocircconn", hc,
                                             pre, post);
   PV::Example * pre2 = new PV::Example("test_cocirc pre 2", hc);
   PV::Example * post2 = new PV::Example("test_cocirc post 2", hc);
   PV::HyPerConn * cHyPer1to2 = new HyPerConn("test_cocirc hyperconn 1 to 2", hc, pre,
                                              post2);
   PV::KernelConn * cCocirc1to2 = new KernelConn("test_cocirc cocircconn 1 to 2", hc,
                                                 pre, post2);
   PV::HyPerConn * cHyPer2to1 = new HyPerConn("test_cocirc hyperconn 2 to 1", hc, pre2,
                                              post);
   PV::KernelConn * cCocirc2to1 = new KernelConn("test_cocirc cocircconn 2 to 1", hc,
                                                 pre2, post);

   const int axonID = 0;
   int num_pre_extended = pre->clayer->numExtended;
   assert(num_pre_extended == cHyPer->getNumWeightPatches());

   int status = 0;
   for (int kPre = 0; kPre < num_pre_extended; kPre++) {
      status = check_cocirc_vs_hyper(cHyPer, cCocirc, kPre, axonID);
      assert(status==0);
      status = check_cocirc_vs_hyper(cHyPer1to2, cCocirc1to2, kPre, axonID);
      assert(status==0);
      status = check_cocirc_vs_hyper(cHyPer2to1, cCocirc2to1, kPre, axonID);
      assert(status==0);
   }

   delete hc;
   return 0;
}

int check_cocirc_vs_hyper(HyPerConn * cHyPer, KernelConn * cKernel, int kPre, int axonID)
{
   int status = 0;
   PVPatch * hyperPatch = cHyPer->getWeights(kPre, axonID);
   PVPatch * cocircPatch = cKernel->getWeights(kPre, axonID);
   int hyPerDataIndex = cHyPer->patchIndexToDataIndex(kPre);
   int kernelDataIndex = cKernel->patchIndexToDataIndex(kPre);
   
   int nk = cHyPer->fPatchSize() * (int) hyperPatch->nx; //; hyperPatch->nf * hyperPatch->nx;
   assert(nk == (cKernel->fPatchSize() * (int) cocircPatch->nx)); // assert(nk == (cocircPatch->nf * cocircPatch->nx));
   int ny = hyperPatch->ny;
   assert(ny == cocircPatch->ny);
   int sy = cHyPer->yPatchStride(); // hyperPatch->sy;
   assert(sy == cKernel->yPatchStride()); // assert(sy == cocircPatch->sy);
   pvdata_t * hyperWeights = cHyPer->get_wData(axonID, hyPerDataIndex); // hyperPatch->data;
   pvdata_t * cocircWeights = cKernel->get_wDataHead(axonID, kernelDataIndex)+hyperPatch->offset; // cocircPatch->data;
   float test_cond = 0.0f;
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         test_cond = cocircWeights[k] - hyperWeights[k];
         if (fabs(test_cond) > 0.001f) {
            const char * cHyper_filename = "cocirc_hyper.txt";
            cHyPer->writeTextWeights(cHyper_filename, kPre);
            const char * cKernel_filename = "cocirc_cocirc.txt";
            cKernel->writeTextWeights(cKernel_filename, kPre);
         }
         assert(fabs(test_cond) <= 0.001f);
      }
      // advance pointers in y
      hyperWeights += sy;
      cocircWeights += sy;
   }
   return status;
}

