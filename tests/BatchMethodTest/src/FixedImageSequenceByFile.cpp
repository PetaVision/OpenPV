#include "FixedImageSequenceByFile.hpp"

FixedImageSequenceByFile::FixedImageSequenceByFile(
      char const *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   initialize(name, params, comm);
}

void FixedImageSequenceByFile::initialize(
      char const *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   FixedImageSequence::initialize(name, params, comm);
}

void FixedImageSequenceByFile::defineImageSequence() {
   int globalBatchSize = getMPIBlock()->getGlobalBatchDimension() * getLayerLoc()->nbatch;
   mIndexStart         = 0;
   mIndexStepBatch     = 1;
   mIndexStepTime      = globalBatchSize;
}
