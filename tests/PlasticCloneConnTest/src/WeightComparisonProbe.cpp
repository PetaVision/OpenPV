/*
 * ColProbe.cpp
 *
 *  Created on: Mar 26, 2017
 *      Author: pschultz
 */

#include "WeightComparisonProbe.hpp"
#include <cstring>
#include <delivery/HyPerDeliveryFacade.hpp>

namespace PV {

WeightComparisonProbe::WeightComparisonProbe(char const *name, PV::HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

WeightComparisonProbe::~WeightComparisonProbe() {}

int WeightComparisonProbe::initialize_base() { return PV_SUCCESS; }

int WeightComparisonProbe::initialize(char const *name, PV::HyPerCol *hc) {
   return ColProbe::initialize(name, hc);
}

Response::Status WeightComparisonProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   mConnectionList.push_back(message->lookup<HyPerConn>(std::string("ConnA")));
   mConnectionList.push_back(message->lookup<HyPerConn>(std::string("ConnB")));
   mConnectionList.push_back(message->lookup<HyPerConn>(std::string("ConnC")));
   mConnectionList.push_back(message->lookup<HyPerConn>(std::string("ConnD")));

   for (auto &c : mConnectionList) {
      if (!c->getInitInfoCommunicatedFlag()) {
         return Response::POSTPONE;
      }
      auto *deliveryComponent = c->getComponentByType<HyPerDeliveryFacade>();
      pvAssert(deliveryComponent);
      auto *weightsPair = c->getComponentByType<WeightsPair>();
      pvAssert(weightsPair);
      bool deliverPostPerspective = deliveryComponent->getUpdateGSynFromPostPerspective();
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
      if (initialized) {
         FatalIf(
               mNumArbors != c->getNumAxonalArbors(),
               "%s and %s have different numbers of arbors.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               numPatches != c->getNumDataPatches(),
               "%s and %s have different numbers of data patches.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               nxp != c->getPatchSizeX(),
               "%s and %s have different nxp.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               nyp != c->getPatchSizeY(),
               "%s and %s have different nyp.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               nfp != c->getPatchSizeF(),
               "%s and %s have different nfp.\n",
               firstConn.c_str(),
               c->getDescription_c());
      }
      else {
         firstConn          = c->getDescription();
         mNumArbors         = c->getNumAxonalArbors();
         numPatches         = c->getNumDataPatches();
         nxp                = c->getPatchSizeX();
         nyp                = c->getPatchSizeY();
         nfp                = c->getPatchSizeF();
         mNumWeightsInArbor = (std::size_t)(nxp * nyp * nfp * numPatches);
         initialized        = true;
      }
   }
   return Response::SUCCESS;
}

Response::Status WeightComparisonProbe::outputState(double timestamp) {
   for (auto &c : mConnectionList) {
      for (int a = 0; a < mNumArbors; a++) {
         float *firstConn = mConnectionList[0]->getWeightsDataStart(a);
         float *thisConn  = c->getWeightsDataStart(a);
         FatalIf(
               memcmp(firstConn, thisConn, sizeof(float) * mNumWeightsInArbor) != 0,
               "%s and %s do not have the same weights.\n",
               c->getDescription_c(),
               mConnectionList[0]->getDescription_c());
      }
   }
   return Response::SUCCESS;
}

} // end namespace PV
