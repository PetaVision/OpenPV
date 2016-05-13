/**
 * This file tests weight initialization based on cocircularity
 * Test compares HyPerConn to CocircConn,
 * assumes CocircConn produces correct weights
 *
 */

#undef DEBUG_PRINT

#include "Example.hpp"
#include <layers/HyPerLayer.hpp>
#include <connections/HyPerConn.hpp>
#include <io/io.h>
#include <assert.h>

using namespace PV;

// First argument to check_cocirc_vs_hyper should have sharedWeights = false
// Second argument should have sharedWeights = true
int check_cocirc_vs_hyper(HyPerConn * cHyPer, HyPerConn * cKernel, int kPre,
		int axonID);

int main(int argc, char * argv[])
{
   PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   initObj->initialize();
   PV::HyPerCol * hc = new PV::HyPerCol("test_cocirc column", initObj);
   
   const char * preLayerName = "test_cocirc pre";
   const char * postLayerName = "test_cocirc post";
   
   PV::Example * pre = new PV::Example(preLayerName, hc);
   assert(pre);
   PV::Example * post = new PV::Example(postLayerName, hc);
   assert(post);
   PV::HyPerConn * cHyPer = new HyPerConn("test_cocirc hyperconn", hc);
   assert(cHyPer);
   PV::HyPerConn * cCocirc = new HyPerConn("test_cocirc cocircconn", hc);
   assert(cCocirc);
   
   PV::Example * pre2 = new PV::Example("test_cocirc pre 2", hc);
   assert(pre2);
   PV::Example * post2 = new PV::Example("test_cocirc post 2", hc);
   assert(post2);
   PV::HyPerConn * cHyPer1to2 = new HyPerConn("test_cocirc hyperconn 1 to 2", hc);
   assert(cHyPer1to2);
   PV::HyPerConn * cCocirc1to2 = new HyPerConn("test_cocirc cocircconn 1 to 2", hc);
   assert(cCocirc1to2);
   PV::HyPerConn * cHyPer2to1 = new HyPerConn("test_cocirc hyperconn 2 to 1", hc);
   assert(cHyPer2to1);
   PV::HyPerConn * cCocirc2to1 = new HyPerConn("test_cocirc cocircconn 2 to 1", hc);
   assert(cCocirc2to1);

   hc->ensureDirExists(hc->getOutputPath());
   
   for (int l=0; l<hc->numberOfLayers(); l++) {
      HyPerLayer * layer = hc->getLayer(l);
      int status = layer->communicateInitInfo();
      assert(status==PV_SUCCESS);
      layer->setInitInfoCommunicatedFlag();
   }   
   for (int c=0; c<hc->numberOfConnections(); c++) {
      BaseConnection * conn = hc->getConnection(c);
      int status = conn->communicateInitInfo();
      assert(status==PV_SUCCESS);
      conn->setInitInfoCommunicatedFlag();
   }
   
   for (int l=0; l<hc->numberOfLayers(); l++) {
      HyPerLayer * layer = hc->getLayer(l);
      int status = layer->allocateDataStructures();
      assert(status==PV_SUCCESS);
      layer->setDataStructuresAllocatedFlag();
   }
   
   for (int c=0; c<hc->numberOfConnections(); c++) {
      BaseConnection * conn = hc->getConnection(c);
      int status = conn->allocateDataStructures();
      assert(status==PV_SUCCESS);
      conn->setDataStructuresAllocatedFlag();
   }

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
   delete initObj;
   return 0;
}

int check_cocirc_vs_hyper(HyPerConn * cHyPer, HyPerConn * cKernel, int kPre, int axonID)
{
   assert(cKernel->usingSharedWeights()==true);
   assert(cHyPer->usingSharedWeights()==false);
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
   pvwdata_t * hyperWeights = cHyPer->get_wData(axonID, hyPerDataIndex); // hyperPatch->data;
   pvwdata_t * cocircWeights = cKernel->get_wDataHead(axonID, kernelDataIndex)+hyperPatch->offset; // cocircPatch->data;
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

