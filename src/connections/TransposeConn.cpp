/* TransposeConn.cpp
 *
 * Created on: May 16, 2011
 *     Author: peteschultz
 */

#include "TransposeConn.hpp"
#include "components/DependentArborList.hpp"
#include "components/DependentSharedWeights.hpp"
#include "components/TransposePatchSize.hpp"
#include "components/TransposeWeightsPair.hpp"

namespace PV {

TransposeConn::TransposeConn(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

TransposeConn::TransposeConn() {}

TransposeConn::~TransposeConn() {}

void TransposeConn::initialize(char const *name, PVParams *params, Communicator const *comm) {
   HyPerConn::initialize(name, params, comm);
}

void TransposeConn::fillComponentTable() {
   HyPerConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

ArborList *TransposeConn::createArborList() {
   return new DependentArborList(name, parameters(), mCommunicator);
}

PatchSize *TransposeConn::createPatchSize() {
   return new TransposePatchSize(name, parameters(), mCommunicator);
}

SharedWeights *TransposeConn::createSharedWeights() {
   return new DependentSharedWeights(name, parameters(), mCommunicator);
}

WeightsPairInterface *TransposeConn::createWeightsPair() {
   return new TransposeWeightsPair(name, parameters(), mCommunicator);
}

InitWeights *TransposeConn::createWeightInitializer() { return nullptr; }

NormalizeBase *TransposeConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *TransposeConn::createWeightUpdater() { return nullptr; }

OriginalConnNameParam *TransposeConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(name, parameters(), mCommunicator);
}

Response::Status
TransposeConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return Response::NO_ACTION;
}

} // namespace PV
