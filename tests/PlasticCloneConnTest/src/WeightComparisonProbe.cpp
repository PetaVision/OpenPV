/*
 * ColProbe.cpp
 *
 *  Created on: Mar 26, 2017
 *      Author: pschultz
 */

#include "WeightComparisonProbe.hpp"

#include <cstring>

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

int WeightComparisonProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   mConnectionList.push_back(message->lookup<HyPerConn>(std::string("ConnA")));
   mConnectionList.push_back(message->lookup<HyPerConn>(std::string("ConnB")));
   mConnectionList.push_back(message->lookup<HyPerConn>(std::string("ConnC")));
   mConnectionList.push_back(message->lookup<HyPerConn>(std::string("ConnD")));
   return PV_SUCCESS;
}

int WeightComparisonProbe::allocateDataStructures() {
   std::string firstConn;
   int nxp, nyp, nfp, numPatches;
   bool initialized = false;
   for (auto &c : mConnectionList) {
      if (initialized) {
         FatalIf(
               mNumArbors != c->numberOfAxonalArborLists(),
               "%s and %s have different numbers of arbors.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               numPatches != c->getNumDataPatches(),
               "%s and %s have different numbers of data patches.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               nxp != c->xPatchSize(),
               "%s and %s have different nxp.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               nyp != c->yPatchSize(),
               "%s and %s have different nyp.\n",
               firstConn.c_str(),
               c->getDescription_c());
         FatalIf(
               nfp != c->fPatchSize(),
               "%s and %s have different nfp.\n",
               firstConn.c_str(),
               c->getDescription_c());
      }
      else {
         firstConn          = c->getDescription();
         mNumArbors         = c->numberOfAxonalArborLists();
         numPatches         = c->getNumDataPatches();
         nxp                = c->xPatchSize();
         nyp                = c->yPatchSize();
         nfp                = c->fPatchSize();
         mNumWeightsInArbor = (std::size_t)(nxp * nyp * nfp * numPatches);
         initialized        = true;
      }
   }
   return PV_SUCCESS;
}

int WeightComparisonProbe::outputState(double timestamp) {
   for (auto &c : mConnectionList) {
      for (int a = 0; a < mNumArbors; a++) {
         float *firstConn = mConnectionList[0]->get_wDataStart(a);
         float *thisConn  = c->get_wDataStart(a);
         FatalIf(
               memcmp(firstConn, thisConn, sizeof(float) * mNumWeightsInArbor) != 0,
               "%s and %s do not have the same weights.\n",
               c->getDescription_c(),
               mConnectionList[0]->getDescription_c());
      }
   }
   return PV_SUCCESS;
}

} // end namespace PV
