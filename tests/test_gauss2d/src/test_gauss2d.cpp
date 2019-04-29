/**
 * This file tests weight initialization to a 2D Gaussian with sigma = 1.0 and normalized to 1.0
 * Test compares behavior of sharedWeights=false to that of sharedWeights=true.
 * assumes sharedWeights=true (the old kernelconn class) produces correct 2D Gaussian weights
 */

#undef DEBUG_PRINT

#include "columns/ComponentBasedObject.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/PV_Init.hpp"
#include "layers/HyPerLayer.hpp"
#include <components/PatchSize.hpp>
#include <components/SharedWeights.hpp>
#include <components/WeightsPair.hpp>
#include <utils/PVLog.hpp>

using namespace PV;

void broadcastMessage(
      ObserverTable const &objectTable,
      std::shared_ptr<BaseMessage const> messagePtr);

int check_kernel_vs_hyper(
      ComponentBasedObject *cHyPer,
      ComponentBasedObject *cKernel,
      int kPre,
      int axonID);

int main(int argc, char *argv[]) {
   PV_Init *initObj = new PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);
   if (initObj->getParams() == nullptr) {
      initObj->setParams("input/test_gauss2d.params");
   }

   char const *pre_layer_name   = "test_gauss2d_pre";
   char const *post_layer_name  = "test_gauss2d_post";
   char const *pre2_layer_name  = "test_gauss2d_pre2";
   char const *post2_layer_name = "test_gauss2d_post2";
   char const *hyper1to1_name   = "test_gauss2d_hyperconn";
   char const *kernel1to1_name  = "test_gauss2d_kernelconn";
   char const *hyper1to2_name   = "test_gauss2d_hyperconn";
   char const *kernel1to2_name  = "test_gauss2d_kernelconn";
   char const *hyper2to1_name   = "test_gauss2d_hyperconn";
   char const *kernel2to1_name  = "test_gauss2d_kernelconn";

   PV::HyPerCol *hc    = new PV::HyPerCol(initObj);
   PV::HyPerLayer *pre = dynamic_cast<PV::HyPerLayer *>(hc->getObjectFromName(pre_layer_name));
   FatalIf(!pre, "Test failed.\n");
   PV::HyPerLayer *post = dynamic_cast<PV::HyPerLayer *>(hc->getObjectFromName(post_layer_name));
   FatalIf(!post, "Test failed.\n");

   PV::ComponentBasedObject *cHyPer =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(hyper1to1_name));

   PV::ComponentBasedObject *cKernel =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(kernel1to1_name));

   PV::HyPerLayer *pre2 = dynamic_cast<PV::HyPerLayer *>(hc->getObjectFromName(pre2_layer_name));
   FatalIf(!pre2, "Test failed.\n");
   PV::HyPerLayer *post2 = dynamic_cast<PV::HyPerLayer *>(hc->getObjectFromName(post2_layer_name));
   FatalIf(!post2, "Test failed.\n");

   PV::ComponentBasedObject *cHyPer1to2 =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(hyper1to2_name));
   FatalIf(!cHyPer1to2, "Test failed.\n");

   PV::ComponentBasedObject *cKernel1to2 =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(kernel1to2_name));
   FatalIf(!cKernel1to2, "Test failed.\n");

   PV::ComponentBasedObject *cHyPer2to1 =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(hyper2to1_name));
   FatalIf(!cHyPer2to1, "Test failed.\n");

   PV::ComponentBasedObject *cKernel2to1 =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(kernel2to1_name));
   FatalIf(!cKernel2to1, "Test failed.\n");

   int status = PV_SUCCESS;

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

   for (int kPre = 0; kPre < numPreExtended; kPre++) {
      status = check_kernel_vs_hyper(cHyPer, cKernel, kPre, axonID);
      FatalIf(status != PV_SUCCESS, "Test failed.\n");
      status = check_kernel_vs_hyper(cHyPer1to2, cKernel1to2, kPre, axonID);
      FatalIf(status != PV_SUCCESS, "Test failed.\n");
      status = check_kernel_vs_hyper(cHyPer2to1, cKernel2to1, kPre, axonID);
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

int check_kernel_vs_hyper(
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
   Patch const &kernelPatch = kernelPreWeights->getPatch(kPre);
   int hyPerDataIndex       = hyperPreWeights->calcDataIndexFromPatchIndex(kPre);
   int kernelDataIndex      = kernelPreWeights->calcDataIndexFromPatchIndex(kPre);

   auto hyperPatchSize  = cHyPer->getComponentByType<PatchSize>();
   auto kernelPatchSize = cKernel->getComponentByType<PatchSize>();
   int nk               = hyperPatchSize->getPatchSizeF() * (int)hyperPatch.nx;
   FatalIf(nk != (kernelPatchSize->getPatchSizeF() * (int)kernelPatch.nx), "Test failed.\n");
   int ny = hyperPatch.ny;
   FatalIf(ny != kernelPatch.ny, "Test failed.\n");
   int sy = hyperPreWeights->getPatchStrideY();
   FatalIf(sy != kernelPreWeights->getPatchStrideY(), "Test failed.\n");
   float *hyperWeights = hyperPreWeights->getDataFromPatchIndex(axonID, hyPerDataIndex)
                         + hyperPreWeights->getPatch(hyPerDataIndex).offset;
   float *kernelWeights =
         kernelPreWeights->getDataFromDataIndex(axonID, kernelDataIndex) + hyperPatch.offset;
   float test_cond = 0.0f;
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         test_cond = kernelWeights[k] - hyperWeights[k];
         if (std::abs(test_cond) > 0.001f) {
            Fatal(errorMessage);
            errorMessage.printf("y %d\n", y);
            errorMessage.printf("k %d\n", k);
            errorMessage.printf("kernelweight %f\n", (double)kernelWeights[k]);
            errorMessage.printf("hyperWeights %f\n", (double)hyperWeights[k]);
            status = PV_FAILURE;
         }
      }
      // advance pointers in y
      hyperWeights += sy;
      kernelWeights += sy;
   }
   return status;
}
