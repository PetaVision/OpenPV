/**
 * This file tests weight initialization based on cocircularity
 * Test compares HyPerConn to CocircConn,
 * assumes CocircConn produces correct weights
 *
 */

#undef DEBUG_PRINT

#include <columns/ComponentBasedObject.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <components/PatchSize.hpp>
#include <components/SharedWeights.hpp>
#include <components/WeightsPair.hpp>
#include <layers/HyPerLayer.hpp>
#include <observerpattern/ObserverTable.hpp>
#include <utils/PVLog.hpp>

using namespace PV;

void broadcastMessage(
      ObserverTable const &objectTable,
      std::shared_ptr<BaseMessage const> messagePtr);

// First argument to check_cocirc_vs_hyper should have sharedWeights = false
// Second argument should have sharedWeights = true
int check_cocirc_vs_hyper(
      ComponentBasedObject *cHyPer,
      ComponentBasedObject *cKernel,
      int kPre,
      int axonID);

int main(int argc, char *argv[]) {
   PV_Init *initObj = new PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);
   PV::HyPerCol *hc = new PV::HyPerCol(initObj);

   const char *preLayerName  = "test_cocirc_pre";
   const char *postLayerName = "test_cocirc_post";

   PV::HyPerLayer *pre = dynamic_cast<HyPerLayer *>(hc->getObjectFromName(preLayerName));
   FatalIf(!pre, "No layer \"%s\" in the hierarchy.\n", preLayerName);
   PV::HyPerLayer *post = dynamic_cast<HyPerLayer *>(hc->getObjectFromName(postLayerName));
   FatalIf(!post, "No layer \"%s\" in the hierarchy.\n", postLayerName);
   PV::ComponentBasedObject *cHyPer =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("test_cocirc_hyperconn"));
   FatalIf(!cHyPer, "Test failed.\n");
   PV::ComponentBasedObject *cCocirc =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("test_cocirc_cocircconn"));
   FatalIf(!cCocirc, "Test failed.\n");

   PV::HyPerLayer *pre2 = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("test_cocirc_pre2"));
   FatalIf(!pre2, "Test failed.\n");
   PV::HyPerLayer *post2 = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("test_cocirc_post2"));
   FatalIf(!post2, "Test failed.\n");
   PV::ComponentBasedObject *cHyPer1to2 =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("test_cocirc_hyperconn1to2"));
   FatalIf(!cHyPer1to2, "Test failed.\n");
   PV::ComponentBasedObject *cCocirc1to2 =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("test_cocirc_cocircconn1to2"));
   FatalIf(!cCocirc1to2, "Test failed.\n");
   PV::ComponentBasedObject *cHyPer2to1 =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("test_cocirc_hyperconn2to1"));
   FatalIf(!cHyPer2to1, "Test failed.\n");
   PV::ComponentBasedObject *cCocirc2to1 =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("test_cocirc_cocircconn2to1"));
   FatalIf(!cCocirc2to1, "Test failed.\n");

   ensureDirExists(hc->getCommunicator()->getLocalMPIBlock(), hc->getOutputPath());

   auto objectTable = hc->getAllObjectsFlat();

   auto communicateMessagePtr = std::make_shared<CommunicateInitInfoMessage>(
         &objectTable,
         hc->getDeltaTime(),
         hc->getNxGlobal(),
         hc->getNyGlobal(),
         hc->getNBatchGlobal(),
         hc->getNumThreads());
   broadcastMessage(objectTable, communicateMessagePtr);

   auto allocateMessagePtr = std::make_shared<AllocateDataStructuresMessage>();
   broadcastMessage(objectTable, allocateMessagePtr);

   auto initializeMessagePtr = std::make_shared<InitializeStateMessage>(hc->getDeltaTime());
   broadcastMessage(objectTable, initializeMessagePtr);

   const int axonID      = 0;
   int numPreExtended    = pre->getNumExtended();
   auto *hyperPreWeights = cHyPer->getComponentByType<WeightsPair>()->getPreWeights();
   FatalIf(numPreExtended != hyperPreWeights->getGeometry()->getNumPatches(), "Test failed.\n");

   int status = PV_SUCCESS;
   for (int kPre = 0; kPre < numPreExtended; kPre++) {
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
      ObserverTable const &objectTable,
      std::shared_ptr<BaseMessage const> messagePtr) {
   int maxcount = 0;
   Response::Status status;
   do {
      status = Response::SUCCESS;
      for (auto *obj : objectTable) {
         status = status + obj->respond(messagePtr);
      }
      maxcount++;
   } while (status != Response::SUCCESS and maxcount < 10);
   FatalIf(
         status != Response::SUCCESS,
         "broadcastMessage(\"%s\") failed.\n",
         messagePtr->getMessageType().c_str());
}

int check_cocirc_vs_hyper(
      ComponentBasedObject *cHyPer,
      ComponentBasedObject *cKernel,
      int kPre,
      int axonID) {
   FatalIf(
         cKernel->getComponentByType<SharedWeights>()->getSharedWeights() != true,
         "%s should have sharedWeights true.\n",
         cKernel->getDescription_c());
   FatalIf(
         cHyPer->getComponentByType<SharedWeights>()->getSharedWeights() != false,
         "%s should have sharedWeights false.\n",
         cHyPer->getDescription_c());
   int status               = PV_SUCCESS;
   auto *hyperWeightsPair   = cHyPer->getComponentByType<WeightsPair>();
   auto *hyperPreWeights    = hyperWeightsPair->getPreWeights();
   auto *kernelWeightsPair  = cKernel->getComponentByType<WeightsPair>();
   auto *kernelPreWeights   = kernelWeightsPair->getPreWeights();
   Patch const &hyperPatch  = hyperPreWeights->getPatch(kPre);
   Patch const &cocircPatch = kernelPreWeights->getPatch(kPre);
   int hyPerDataIndex       = hyperPreWeights->calcDataIndexFromPatchIndex(kPre);
   int kernelDataIndex      = kernelPreWeights->calcDataIndexFromPatchIndex(kPre);

   auto hyperPatchSize  = cHyPer->getComponentByType<PatchSize>();
   auto kernelPatchSize = cKernel->getComponentByType<PatchSize>();
   int nk               = hyperPatchSize->getPatchSizeF() * (int)hyperPatch.nx;
   FatalIf(nk != (kernelPatchSize->getPatchSizeF() * (int)cocircPatch.nx), "Test failed.\n");
   int ny = hyperPatch.ny;
   FatalIf(ny != cocircPatch.ny, "Test failed.\n");
   int sy = hyperPreWeights->getPatchStrideY();
   FatalIf(sy != kernelPreWeights->getPatchStrideY(), "Test failed.\n");
   float *hyperWeights = hyperPreWeights->getDataFromPatchIndex(axonID, hyPerDataIndex)
                         + hyperPreWeights->getPatch(hyPerDataIndex).offset;
   float *cocircWeights =
         kernelPreWeights->getDataFromDataIndex(axonID, kernelDataIndex) + hyperPatch.offset;
   float test_cond = 0.0f;
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
