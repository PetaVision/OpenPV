#include "FixedImageSequenceByList.hpp"

FixedImageSequenceByList::FixedImageSequenceByList(
      char const *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   initialize(name, params, comm);
}

void FixedImageSequenceByList::initialize(
      char const *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   FixedImageSequence::initialize(name, params, comm);
}

void FixedImageSequenceByList::defineImageSequence() {
   int globalBatchSize = getMPIBlock()->getGlobalBatchDimension() * getLayerLoc()->nbatch;
   mIndexStart         = 0;
   mIndexStepBatch     = mNumImages / globalBatchSize; // integer division
   mIndexStepTime      = 1;
}
