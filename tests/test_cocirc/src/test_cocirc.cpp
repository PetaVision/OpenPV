/**
 * This file tests weight initialization based on cocircularity
 * Test compares HyPerConn to CocircConn,
 * assumes CocircConn produces correct weights
 *
 */

#undef DEBUG_PRINT

#include <connections/HyPerConn.hpp>
#include <io/io.hpp>
#include <layers/HyPerLayer.hpp>
#include <utils/PVLog.hpp>

using namespace PV;

// First argument to check_cocirc_vs_hyper should have sharedWeights = false
// Second argument should have sharedWeights = true
int check_cocirc_vs_hyper(HyPerConn *cHyPer, HyPerConn *cKernel, int kPre, int axonID);

int main(int argc, char *argv[]) {
   PV_Init *initObj = new PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);
   PV::HyPerCol *hc = new PV::HyPerCol(initObj);

   const char *preLayerName  = "test_cocirc_pre";
   const char *postLayerName = "test_cocirc_post";

   PV::HyPerLayer *pre = dynamic_cast<HyPerLayer *>(hc->getObjectFromName(preLayerName));
   FatalIf(!pre, "No layer \"%s\" in the hierarchy.\n", preLayerName);
   PV::HyPerLayer *post = dynamic_cast<HyPerLayer *>(hc->getObjectFromName(postLayerName));
   FatalIf(!post, "No layer \"%s\" in the hierarchy.\n", postLayerName);
   PV::HyPerConn *cHyPer =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_hyperconn"));
   FatalIf(!cHyPer, "Test failed.\n");
   PV::HyPerConn *cCocirc =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_cocircconn"));
   FatalIf(!(cCocirc), "Test failed.\n");

   PV::HyPerLayer *pre2 = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("test_cocirc_pre2"));
   FatalIf(!(pre2), "Test failed.\n");
   PV::HyPerLayer *post2 = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("test_cocirc_post2"));
   FatalIf(!(post2), "Test failed.\n");
   PV::HyPerConn *cHyPer1to2 =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_hyperconn1to2"));
   FatalIf(!(cHyPer1to2), "Test failed.\n");
   PV::HyPerConn *cCocirc1to2 =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_cocircconn1to2"));
   FatalIf(!(cCocirc1to2), "Test failed.\n");
   PV::HyPerConn *cHyPer2to1 =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_hyperconn2to1"));
   FatalIf(!(cHyPer2to1), "Test failed.\n");
   PV::HyPerConn *cCocirc2to1 =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_cocircconn2to1"));
   FatalIf(!(cCocirc2to1), "Test failed.\n");

   ensureDirExists(hc->getCommunicator()->getLocalMPIBlock(), hc->getOutputPath());

   auto objectMap      = hc->copyObjectMap();
   auto commMessagePtr = std::make_shared<CommunicateInitInfoMessage>(*objectMap);
   for (Observer *obj = hc->getNextObject(nullptr); obj != nullptr; obj = hc->getNextObject(obj)) {
      int status = obj->respond(commMessagePtr);
      FatalIf(status != PV_SUCCESS, "Test failed.\n");
   }

   auto allocateMessagePtr = std::make_shared<AllocateDataMessage>();
   for (Observer *obj = hc->getNextObject(nullptr); obj != nullptr; obj = hc->getNextObject(obj)) {
      int status = obj->respond(allocateMessagePtr);
      FatalIf(status != PV_SUCCESS, "Test failed.\n");
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
            cHyPer->writeTextWeights(cHyper_filename, false /*verifyWrites*/, kPre);
            const char *cKernel_filename = "cocirc_cocirc.txt";
            cKernel->writeTextWeights(cKernel_filename, false /*verifyWrites*/, kPre);
         }
         FatalIf(!(fabsf(test_cond) <= 0.001f), "Test failed.\n");
      }
      // advance pointers in y
      hyperWeights += sy;
      cocircWeights += sy;
   }
   return status;
}
