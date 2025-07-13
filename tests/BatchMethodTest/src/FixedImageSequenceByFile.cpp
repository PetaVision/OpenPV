#include "FixedImageSequenceByFile.hpp"

FixedImageSequenceByFile::FixedImageSequenceByFile(
      std::shared_ptr<PV::ParamGroup> params,
      std::shared_ptr<PV::ParamGroup> defaults,
      PV::Communicator const *comm) {
   initialize(params, defaults, comm);
}

void FixedImageSequenceByFile::initialize(
      std::shared_ptr<PV::ParamGroup> params,
      std::shared_ptr<PV::ParamGroup> defaults,
      PV::Communicator const *comm) {
   FixedImageSequence::initialize(params, defaults, comm);
}

void FixedImageSequenceByFile::defineImageSequence() {
   int globalBatchDim  = getCommunicator()->getIOMPIBlock()->getGlobalBatchDimension();
   int globalBatchSize = globalBatchDim * getLayerLoc()->nbatch;
   mIndexStart         = 0;
   mIndexStepBatch     = 1;
   mIndexStepTime      = globalBatchSize;
}
