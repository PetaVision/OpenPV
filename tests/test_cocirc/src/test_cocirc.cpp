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

void broadcastMessage(
      std::map<std::string, Observer *> *objectMap,
      std::shared_ptr<BaseMessage const> messagePtr);

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
   FatalIf(!cCocirc, "Test failed.\n");

   PV::HyPerLayer *pre2 = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("test_cocirc_pre2"));
   FatalIf(!pre2, "Test failed.\n");
   PV::HyPerLayer *post2 = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("test_cocirc_post2"));
   FatalIf(!post2, "Test failed.\n");
   PV::HyPerConn *cHyPer1to2 =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_hyperconn1to2"));
   FatalIf(!cHyPer1to2, "Test failed.\n");
   PV::HyPerConn *cCocirc1to2 =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_cocircconn1to2"));
   FatalIf(!cCocirc1to2, "Test failed.\n");
   PV::HyPerConn *cHyPer2to1 =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_hyperconn2to1"));
   FatalIf(!cHyPer2to1, "Test failed.\n");
   PV::HyPerConn *cCocirc2to1 =
         dynamic_cast<HyPerConn *>(hc->getObjectFromName("test_cocirc_cocircconn2to1"));
   FatalIf(!cCocirc2to1, "Test failed.\n");

   ensureDirExists(hc->getCommunicator()->getLocalMPIBlock(), hc->getOutputPath());

   auto objectMap = hc->copyObjectMap();

   auto communicateMessagePtr = std::make_shared<CommunicateInitInfoMessage>(*objectMap);
   broadcastMessage(objectMap, communicateMessagePtr);

   auto allocateMessagePtr = std::make_shared<AllocateDataMessage>();
   broadcastMessage(objectMap, allocateMessagePtr);

   auto initializeMessagePtr = std::make_shared<InitializeStateMessage>();
   broadcastMessage(objectMap, initializeMessagePtr);

   delete objectMap;

   const int axonID     = 0;
   int num_pre_extended = pre->clayer->numExtended;
   FatalIf(num_pre_extended != cHyPer->getNumGeometryPatches(), "Test failed.\n");

   int status = PV_SUCCESS;
   for (int kPre = 0; kPre < num_pre_extended; kPre++) {
      status = check_cocirc_vs_hyper(cHyPer, cCocirc, kPre, axonID);
      FatalIf(status != PV_SUCCESS, "Test failed.\n");
      status = check_cocirc_vs_hyper(cHyPer1to2, cCocirc1to2, kPre, axonID);
      FatalIf(status != PV_SUCCESS, "Test failed.\n");
      status = check_cocirc_vs_hyper(cHyPer2to1, cCocirc2to1, kPre, axonID);
      FatalIf(status != PV_SUCCESS, "Test failed.\n");
   }

   delete hc;
   delete initObj;
   return status;
}

void broadcastMessage(
      std::map<std::string, Observer *> *objectMap,
      std::shared_ptr<BaseMessage const> messagePtr) {
   int maxcount = 0;
   Response::Status status;
   do {
      status = Response::SUCCESS;
      for (auto &p : *objectMap) {
         Observer *obj = p.second;
         status        = status + obj->respond(messagePtr);
      }
      maxcount++;
   } while (status != Response::SUCCESS and maxcount < 10);
   FatalIf(
         status != Response::SUCCESS,
         "broadcastMessage(\"%s\") failed.\n",
         messagePtr->getMessageType().c_str());
}

int check_cocirc_vs_hyper(HyPerConn *cHyPer, HyPerConn *cKernel, int kPre, int axonID) {
   FatalIf(cKernel->getSharedWeights() != true, "Test failed.\n");
   FatalIf(cHyPer->getSharedWeights() != false, "Test failed.\n");
   int status               = PV_SUCCESS;
   Patch const *hyperPatch  = cHyPer->getPatch(kPre);
   Patch const *cocircPatch = cKernel->getPatch(kPre);
   int hyPerDataIndex       = cHyPer->calcDataIndexFromPatchIndex(kPre);
   int kernelDataIndex      = cKernel->calcDataIndexFromPatchIndex(kPre);

   int nk = cHyPer->getPatchSizeF() * (int)hyperPatch->nx;
   FatalIf(nk != (cKernel->getPatchSizeF() * (int)cocircPatch->nx), "Test failed.\n");
   int ny = hyperPatch->ny;
   FatalIf(ny != cocircPatch->ny, "Test failed.\n");
   int sy = cHyPer->getPatchStrideY();
   FatalIf(sy != cKernel->getPatchStrideY(), "Test failed.\n");
   float *hyperWeights  = cHyPer->getWeightsData(axonID, hyPerDataIndex);
   float *cocircWeights = cKernel->getWeightsDataHead(axonID, kernelDataIndex) + hyperPatch->offset;
   float test_cond      = 0.0f;
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         test_cond = cocircWeights[k] - hyperWeights[k];
         if (std::abs(test_cond) > 0.001f) {
            ErrorLog().printf(
                  "axodID %d, patch index %d, k=%d, y=%d: %s weight is %f; %s is %f.\n",
                  axonID,
                  kPre,
                  k,
                  y,
                  cHyPer->getName(),
                  cKernel->getName());
            status = PV_SUCCESS;
         }
      }
      // advance pointers in y
      hyperWeights += sy;
      cocircWeights += sy;
   }
   FatalIf(status != PV_SUCCESS, "Test failed.\n");
   return status;
}
