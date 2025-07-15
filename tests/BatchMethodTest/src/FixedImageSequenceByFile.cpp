#include "FixedImageSequenceByFile.hpp"

FixedImageSequenceByFile::FixedImageSequenceByFile(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm) {
   initialize(paramsIO, comm);
}

void FixedImageSequenceByFile::initialize(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm) {
   FixedImageSequence::initialize(paramsIO, comm);
}

void FixedImageSequenceByFile::defineImageSequence() {
   int globalBatchDim  = getCommunicator()->getIOMPIBlock()->getGlobalBatchDimension();
   int globalBatchSize = globalBatchDim * getLayerLoc()->nbatch;
   mIndexStart         = 0;
   mIndexStepBatch     = 1;
   mIndexStepTime      = globalBatchSize;
}
