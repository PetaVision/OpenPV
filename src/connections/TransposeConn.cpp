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

TransposeConn::TransposeConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

TransposeConn::TransposeConn() {}

TransposeConn::~TransposeConn() {}

void TransposeConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerConn::initialize(params, defaults, comm);
   mWriteInitializeFromCheckpointFlag = false;
}

void TransposeConn::fillComponentTable() {
   HyPerConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

ArborList *TransposeConn::createArborList() {
   return new DependentArborList(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

PatchSize *TransposeConn::createPatchSize() {
   return new TransposePatchSize(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

SharedWeights *TransposeConn::createSharedWeights() {
   return new DependentSharedWeights(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

WeightsPairInterface *TransposeConn::createWeightsPair() {
   return new TransposeWeightsPair(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

InitWeights *TransposeConn::createWeightInitializer() { return nullptr; }

NormalizeBase *TransposeConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *TransposeConn::createWeightUpdater() { return nullptr; }

OriginalConnNameParam *TransposeConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

Response::Status
TransposeConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return Response::NO_ACTION;
}

} // namespace PV
