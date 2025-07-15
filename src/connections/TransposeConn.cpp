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

TransposeConn::TransposeConn(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

TransposeConn::TransposeConn() {}

TransposeConn::~TransposeConn() {}

void TransposeConn::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerConn::initialize(paramsIO, comm);
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
   return new DependentArborList(mParamsIO, mCommunicator);
}

PatchSize *TransposeConn::createPatchSize() {
   return new TransposePatchSize(mParamsIO, mCommunicator);
}

SharedWeights *TransposeConn::createSharedWeights() {
   return new DependentSharedWeights(mParamsIO, mCommunicator);
}

WeightsPairInterface *TransposeConn::createWeightsPair() {
   return new TransposeWeightsPair(mParamsIO, mCommunicator);
}

InitWeights *TransposeConn::createWeightInitializer() { return nullptr; }

NormalizeBase *TransposeConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *TransposeConn::createWeightUpdater() { return nullptr; }

OriginalConnNameParam *TransposeConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(mParamsIO, mCommunicator);
}

Response::Status
TransposeConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return Response::NO_ACTION;
}

} // namespace PV
