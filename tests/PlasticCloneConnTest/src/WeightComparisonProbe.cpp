/*
 * ColProbe.cpp
 *
 *  Created on: Mar 26, 2017
 *      Author: pschultz
 */

#include "WeightComparisonProbe.hpp"
#include <components/ArborList.hpp>
#include <components/PatchSize.hpp>
#include <components/WeightsPair.hpp>
#include <delivery/HyPerDeliveryCreator.hpp>

#include <cstring>

namespace PV {

WeightComparisonProbe::WeightComparisonProbe(
      char const *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

WeightComparisonProbe::~WeightComparisonProbe() {}

int WeightComparisonProbe::initialize_base() { return PV_SUCCESS; }

void WeightComparisonProbe::initialize(
      char const *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   ColProbe::initialize(name, params, comm);
}

Response::Status WeightComparisonProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable = message->mObjectTable;
   mConnectionList.push_back(objectTable->findObject<ComponentBasedObject>("ConnA"));
   mConnectionList.push_back(objectTable->findObject<ComponentBasedObject>("ConnB"));
   mConnectionList.push_back(objectTable->findObject<ComponentBasedObject>("ConnC"));
   mConnectionList.push_back(objectTable->findObject<ComponentBasedObject>("ConnD"));

   for (auto &c : mConnectionList) {
      if (!c->getInitInfoCommunicatedFlag()) {
         return Response::POSTPONE;
      }
      auto *deliveryCreator = objectTable->findObject<HyPerDeliveryCreator>(c->getName());
      pvAssert(deliveryCreator);
      auto *weightsPair = objectTable->findObject<WeightsPair>(c->getName());
      pvAssert(weightsPair);
      bool deliverPostPerspective = deliveryCreator->getUpdateGSynFromPostPerspective();
      if (deliverPostPerspective) {
         weightsPair->needPost();
      }
      else {
         weightsPair->needPre();
      }
   }
   return Response::SUCCESS;
}

Response::Status WeightComparisonProbe::allocateDataStructures() {
   std::string firstConn;
   int nxp, nyp, nfp, numPatches;
   bool initialized = false;
   for (auto &c : mConnectionList) {
      int const numArbors = c->getComponentByType<ArborList>()->getNumAxonalArbors();
      auto *preWeights    = c->getComponentByType<WeightsPair>()->getPreWeights();
      auto *patchSize     = c->getComponentByType<PatchSize>();
      numPatches          = preWeights->getNumDataPatches();
      nxp                 = patchSize->getPatchSizeX();
      nyp                 = patchSize->getPatchSizeY();
      nfp                 = patchSize->getPatchSizeF();
      if (initialized) {
         FatalIf(
               numArbors != mNumArbors,
               "%s and %s have different numbers of arbors.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               mNumWeightsInArbor != nxp * nyp * nfp * numPatches,
               "%s and %s have different numbers of data weights.\n",
               firstConn.c_str(),
               c->getDescription_c());
         auto *patchSize = c->getComponentByType<PatchSize>();
         FatalIf(
               nxp != patchSize->getPatchSizeX(),
               "%s and %s have different nxp.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               nyp != patchSize->getPatchSizeY(),
               "%s and %s have different nyp.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               nfp != patchSize->getPatchSizeF(),
               "%s and %s have different nfp.\n",
               firstConn.c_str(),
               c->getDescription_c());
      }
      else {
         firstConn          = c->getDescription();
         mNumArbors         = numArbors;
         mNumWeightsInArbor = nxp * nyp * nfp * numPatches;
         initialized        = true;
      }
   }
   return Response::SUCCESS;
}

Response::Status WeightComparisonProbe::outputState(double simTime, double deltaTime) {
   auto *firstConnWeights = mConnectionList[0]->getComponentByType<WeightsPair>()->getPreWeights();
   for (auto &c : mConnectionList) {
      auto *thisConnWeights = c->getComponentByType<WeightsPair>()->getPreWeights();
      for (int a = 0; a < mNumArbors; a++) {
         float *firstConnWeightData = firstConnWeights->getData(a);
         float *thisConnWeightData  = thisConnWeights->getData(a);
         std::size_t memsize        = sizeof(float) * (std::size_t)mNumWeightsInArbor;
         FatalIf(
               memcmp(firstConnWeightData, thisConnWeightData, memsize) != 0,
               "%s and %s do not have the same weights.\n",
               c->getDescription_c(),
               mConnectionList[0]->getDescription_c());
      }
   }
   return Response::SUCCESS;
}

double WeightComparisonProbe::referenceUpdateTime(double simTime) const { return simTime; }

} // end namespace PV
