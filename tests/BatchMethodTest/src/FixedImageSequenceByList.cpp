#include "FixedImageSequenceByList.hpp"

FixedImageSequenceByList::FixedImageSequenceByList(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm) {
   initialize(paramsIO, comm);
}

void FixedImageSequenceByList::initialize(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm) {
   FixedImageSequence::initialize(paramsIO, comm);
}

void FixedImageSequenceByList::defineImageSequence() {
   int globalBatchDim  = getCommunicator()->getIOMPIBlock()->getGlobalBatchDimension();
   int globalBatchSize = globalBatchDim * getLayerLoc()->nbatch;
   mIndexStart         = 0;
   mIndexStepBatch     = mNumImages / globalBatchSize; // integer division
   mIndexStepTime      = 1;
}
