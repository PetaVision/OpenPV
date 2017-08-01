/*
 * NormalizeGroup.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: pschultz
 */

#include <normalizers/NormalizeGroup.hpp>

namespace PV {

NormalizeGroup::NormalizeGroup(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

NormalizeGroup::NormalizeGroup() {}

NormalizeGroup::~NormalizeGroup() { free(normalizeGroupName); }

int NormalizeGroup::initialize_base() { return PV_SUCCESS; }

int NormalizeGroup::initialize(char const *name, HyPerCol *hc) {
   int status = NormalizeBase::initialize(name, hc);
   return status;
}

int NormalizeGroup::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   ioParam_normalizeGroupName(ioFlag);
   return status;
}

// The NormalizeBase parameters are overridden to do nothing in NormalizeGroup.
void NormalizeGroup::ioParam_strength(enum ParamsIOFlag ioFlag) {}
void NormalizeGroup::ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) {}
void NormalizeGroup::ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag) {}
void NormalizeGroup::ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag) {}

void NormalizeGroup::ioParam_normalizeGroupName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "normalizeGroupName", &normalizeGroupName);
}

int NormalizeGroup::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   groupHead = parent->getNormalizerFromName(normalizeGroupName);
   if (groupHead == nullptr) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: normalizeGroupName \"%s\" is not a recognized normalizer.\n",
               getDescription_c(),
               normalizeGroupName);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   HyPerConn *conn = message->lookup<HyPerConn>(std::string(name));
   pvAssertMessage(conn != nullptr, "No connection \"%s\" for %s.\n", name, getDescription_c());
   return groupHead->addConnToList(conn);
}

int NormalizeGroup::normalizeWeights() { return PV_SUCCESS; }

} /* namespace PV */
