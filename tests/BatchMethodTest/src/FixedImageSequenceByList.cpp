#include "FixedImageSequenceByList.hpp"

FixedImageSequenceByList::FixedImageSequenceByList(
      std::shared_ptr<PV::ParamGroup> params,
      std::shared_ptr<PV::ParamGroup> defaults,
      PV::Communicator const *comm) {
   initialize(params, defaults, comm);
}

void FixedImageSequenceByList::initialize(
      std::shared_ptr<PV::ParamGroup> params,
      std::shared_ptr<PV::ParamGroup> defaults,
      PV::Communicator const *comm) {
   FixedImageSequence::initialize(params, defaults, comm);
}

void FixedImageSequenceByList::defineImageSequence() {
   int globalBatchDim  = getCommunicator()->getIOMPIBlock()->getGlobalBatchDimension();
   int globalBatchSize = globalBatchDim * getLayerLoc()->nbatch;
   mIndexStart         = 0;
   mIndexStepBatch     = mNumImages / globalBatchSize; // integer division
   mIndexStepTime      = 1;
}
