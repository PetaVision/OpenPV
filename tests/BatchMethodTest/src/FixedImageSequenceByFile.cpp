#include "FixedImageSequenceByFile.hpp"

FixedImageSequenceByFile::FixedImageSequenceByFile(char const *name, PV::HyPerCol *hc) {
   initialize(name, hc);
}

int FixedImageSequenceByFile::initialize(char const *name, PV::HyPerCol *hc) {
   return FixedImageSequence::initialize(name, hc);
}

void FixedImageSequenceByFile::defineImageSequence() {
   int globalBatchSize = getMPIBlock()->getGlobalBatchDimension() * getLayerLoc()->nbatch;
   mIndexStart         = 0;
   mIndexStepBatch     = 1;
   mIndexStepTime      = globalBatchSize;
}
