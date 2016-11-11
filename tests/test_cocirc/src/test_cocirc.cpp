/**
 * This file tests weight initialization based on cocircularity
 * Test compares HyPerConn to CocircConn,
 * assumes CocircConn produces correct weights
 *
 */

#undef DEBUG_PRINT

#include "Example.hpp"
#include "connections/HyPerConn.hpp"
#include "io/io.hpp"
#include "layers/HyPerLayer.hpp"
#include <utils/PVLog.hpp>

using namespace PV;

// First argument to check_cocirc_vs_hyper should have sharedWeights = false
// Second argument should have sharedWeights = true
int check_cocirc_vs_hyper(HyPerConn *cHyPer, HyPerConn *cKernel, int kPre, int axonID);

int main(int argc, char *argv[]) {
   PV_Init *initObj = new PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);
   PV::HyPerCol *hc = new PV::HyPerCol("test_cocirc column", initObj);

   const char *preLayerName  = "test_cocirc pre";
   const char *postLayerName = "test_cocirc post";

   PV::Example *pre = new PV::Example(preLayerName, hc);
   FatalIf(!(pre), "Test failed.\n");
   PV::Example *post = new PV::Example(postLayerName, hc);
   FatalIf(!(post), "Test failed.\n");
   PV::HyPerConn *cHyPer = new HyPerConn("test_cocirc hyperconn", hc);
   FatalIf(!(cHyPer), "Test failed.\n");
   PV::HyPerConn *cCocirc = new HyPerConn("test_cocirc cocircconn", hc);
   FatalIf(!(cCocirc), "Test failed.\n");

   PV::Example *pre2 = new PV::Example("test_cocirc pre 2", hc);
   FatalIf(!(pre2), "Test failed.\n");
   PV::Example *post2 = new PV::Example("test_cocirc post 2", hc);
   FatalIf(!(post2), "Test failed.\n");
   PV::HyPerConn *cHyPer1to2 = new HyPerConn("test_cocirc hyperconn 1 to 2", hc);
   FatalIf(!(cHyPer1to2), "Test failed.\n");
   PV::HyPerConn *cCocirc1to2 = new HyPerConn("test_cocirc cocircconn 1 to 2", hc);
   FatalIf(!(cCocirc1to2), "Test failed.\n");
   PV::HyPerConn *cHyPer2to1 = new HyPerConn("test_cocirc hyperconn 2 to 1", hc);
   FatalIf(!(cHyPer2to1), "Test failed.\n");
   PV::HyPerConn *cCocirc2to1 = new HyPerConn("test_cocirc cocircconn 2 to 1", hc);
   FatalIf(!(cCocirc2to1), "Test failed.\n");

   ensureDirExists(hc->getCommunicator(), hc->getOutputPath());

   auto objectMap      = hc->copyObjectMap();
   auto commMessagePtr = std::make_shared<CommunicateInitInfoMessage>(*objectMap);
   for (int l = 0; l < hc->numberOfLayers(); l++) {
      HyPerLayer *layer = hc->getLayer(l);
      int status        = layer->respond(commMessagePtr);
      FatalIf(!(status == PV_SUCCESS), "Test failed.\n");
   }
   for (int c = 0; c < hc->numberOfConnections(); c++) {
      BaseConnection *conn = hc->getConnection(c);
      int status           = conn->respond(commMessagePtr);
      FatalIf(!(status == PV_SUCCESS), "Test failed.\n");
   }
   delete objectMap;

   auto allocateMessagePtr = std::make_shared<AllocateDataMessage>();
   for (int l = 0; l < hc->numberOfLayers(); l++) {
      HyPerLayer *layer = hc->getLayer(l);
      int status        = layer->respond(allocateMessagePtr);
      FatalIf(!(status == PV_SUCCESS), "Test failed.\n");
   }

   for (int c = 0; c < hc->numberOfConnections(); c++) {
      BaseConnection *conn = hc->getConnection(c);
      int status           = conn->respond(allocateMessagePtr);
      FatalIf(!(status == PV_SUCCESS), "Test failed.\n");
   }

   const int axonID     = 0;
   int num_pre_extended = pre->clayer->numExtended;
   FatalIf(!(num_pre_extended == cHyPer->getNumWeightPatches()), "Test failed.\n");

   int status = 0;
   for (int kPre = 0; kPre < num_pre_extended; kPre++) {
      status = check_cocirc_vs_hyper(cHyPer, cCocirc, kPre, axonID);
      FatalIf(!(status == 0), "Test failed.\n");
      status = check_cocirc_vs_hyper(cHyPer1to2, cCocirc1to2, kPre, axonID);
      FatalIf(!(status == 0), "Test failed.\n");
      status = check_cocirc_vs_hyper(cHyPer2to1, cCocirc2to1, kPre, axonID);
      FatalIf(!(status == 0), "Test failed.\n");
   }

   delete hc;
   delete initObj;
   return 0;
}

int check_cocirc_vs_hyper(HyPerConn *cHyPer, HyPerConn *cKernel, int kPre, int axonID) {
   FatalIf(!(cKernel->usingSharedWeights() == true), "Test failed.\n");
   FatalIf(!(cHyPer->usingSharedWeights() == false), "Test failed.\n");
   int status           = 0;
   PVPatch *hyperPatch  = cHyPer->getWeights(kPre, axonID);
   PVPatch *cocircPatch = cKernel->getWeights(kPre, axonID);
   int hyPerDataIndex   = cHyPer->patchIndexToDataIndex(kPre);
   int kernelDataIndex  = cKernel->patchIndexToDataIndex(kPre);

   int nk = cHyPer->fPatchSize() * (int)hyperPatch->nx;
   FatalIf(!(nk == (cKernel->fPatchSize() * (int)cocircPatch->nx)), "Test failed.\n");
   int ny = hyperPatch->ny;
   FatalIf(!(ny == cocircPatch->ny), "Test failed.\n");
   int sy = cHyPer->yPatchStride();
   FatalIf(!(sy == cKernel->yPatchStride()), "Test failed.\n");
   float *hyperWeights  = cHyPer->get_wData(axonID, hyPerDataIndex);
   float *cocircWeights = cKernel->get_wDataHead(axonID, kernelDataIndex) + hyperPatch->offset;
   float test_cond      = 0.0f;
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         test_cond = cocircWeights[k] - hyperWeights[k];
         if (fabsf(test_cond) > 0.001f) {
            const char *cHyper_filename = "cocirc_hyper.txt";
            cHyPer->writeTextWeights(cHyper_filename, kPre);
            const char *cKernel_filename = "cocirc_cocirc.txt";
            cKernel->writeTextWeights(cKernel_filename, kPre);
         }
         FatalIf(!(fabsf(test_cond) <= 0.001f), "Test failed.\n");
      }
      // advance pointers in y
      hyperWeights += sy;
      cocircWeights += sy;
   }
   return status;
}
