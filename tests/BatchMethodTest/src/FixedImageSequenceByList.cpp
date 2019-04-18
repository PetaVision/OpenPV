#include "FixedImageSequenceByList.hpp"

FixedImageSequenceByList::FixedImageSequenceByList(char const *name, PV::HyPerCol *hc) {
   initialize(name, hc);
}

int FixedImageSequenceByList::initialize(char const *name, PV::HyPerCol *hc) {
   return FixedImageSequence::initialize(name, hc);
}

void FixedImageSequenceByList::defineImageSequence() {
   int globalBatchSize = getMPIBlock()->getGlobalBatchDimension() * getLayerLoc()->nbatch;
   mIndexStart         = 0;
   mIndexStepBatch     = mNumImages / globalBatchSize; // integer division
   mIndexStepTime      = 1;
}
