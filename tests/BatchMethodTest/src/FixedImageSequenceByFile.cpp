#include "FixedImageSequenceByFile.hpp"

FixedImageSequenceByFile::FixedImageSequenceByFile(
      char const *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   initialize(name, params, comm);
}

void FixedImageSequenceByFile::initialize(
      char const *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   FixedImageSequence::initialize(name, params, comm);
}

void FixedImageSequenceByFile::defineImageSequence() {
   int globalBatchDim  = getCommunicator()->getIOMPIBlock()->getGlobalBatchDimension();
   int globalBatchSize = globalBatchDim * getLayerLoc()->nbatch;
   mIndexStart         = 0;
   mIndexStepBatch     = 1;
   mIndexStepTime      = globalBatchSize;
}
