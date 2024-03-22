#include "FixedImageSequence.hpp"
#include <components/ActivityBuffer.hpp>
#include <components/ActivityComponentActivityOnly.hpp>
#include <structures/Image.hpp>
#include <utils/BufferUtilsMPI.hpp>

using namespace PV;

FixedImageSequence::FixedImageSequence(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *FixedImageSequence::createActivityComponent() {
   return new ActivityComponentActivityOnly<ActivityBuffer>(getName(), parameters(), mCommunicator);
}

Response::Status
FixedImageSequence::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   mActivityPointer =
         mActivityComponent->getComponentByType<ActivityBuffer>()->getReadWritePointer();
   float *A = mActivityPointer;
   for (int k = 0; k < getNumNeuronsAllBatches(); k++) {
      A[k] = 0.0f;
   }
   defineImageSequence();
   return Response::SUCCESS;
}

Response::Status FixedImageSequence::checkUpdateState(double simTime, double deltaTime) {
   FatalIf(deltaTime != 1.0, "FixedImageSequence assumes dt = 1.\n");
   double timestampRounded = std::nearbyint(simTime);
   FatalIf(
         simTime != timestampRounded,
         "FixedImageSequence::checkUpdateState() requires the time argument be an integer.\n");
   PVLayerLoc const *loc = getLayerLoc();
   int timestampInt      = (int)timestampRounded;
   int localNBatch       = loc->nbatch;

   auto ioMPIBlock = getCommunicator()->getIOMPIBlock();
   for (int m = 0; m < ioMPIBlock->getBatchDimension(); m++) {
      int mpiBlockIndex = m + ioMPIBlock->getStartBatch();
      for (int b = 0; b < localNBatch; b++) {
         Buffer<float> buffer;
         if (ioMPIBlock->getRank() == 0) {
            int globalBatchElement = b + localNBatch * mpiBlockIndex;
            int inputIndex         = mIndexStart + (timestampInt - 1) * mIndexStepTime;
            inputIndex += mIndexStepBatch * globalBatchElement;
            inputIndex %= mNumImages;

            auto filename = std::string("input/images/") + std::to_string(inputIndex) + ".png";
            Image image(filename);
            bool sameDims = image.getWidth() == loc->nxGlobal and image.getHeight() == loc->nyGlobal
                            and image.getFeatures() == loc->nf;
            FatalIf(
                  !sameDims,
                  "File \"%s\" is %dx%dx%d, but %s is %dx%dx%d.\n",
                  filename.c_str(),
                  image.getWidth(),
                  image.getHeight(),
                  image.getFeatures(),
                  getDescription_c(),
                  loc->nxGlobal,
                  loc->nyGlobal,
                  loc->nf);
            buffer.set(image);
         }
         else {
            buffer.resize(loc->nx, loc->ny, loc->nf);
         }
         BufferUtils::scatter<float>(ioMPIBlock, buffer, loc->nx, loc->ny, m, 0);
         if (ioMPIBlock->getBatchIndex() != m) {
            continue;
         }
         bool sameDims = buffer.getWidth() == loc->nx and buffer.getHeight() == loc->ny
                         and buffer.getFeatures() == loc->nf;
         FatalIf(
               !sameDims,
               "Image for t=%f scattered to a %dx%dx%d buffer, but local size of %s is %dx%dx%d.\n",
               simTime,
               buffer.getWidth(),
               buffer.getHeight(),
               buffer.getFeatures(),
               getDescription_c(),
               loc->nx,
               loc->ny,
               loc->nf);
         float *activity = &mActivityPointer[b * mActivityComponent->getNumExtended()];
         for (int k = 0; k < getNumNeurons(); k++) {
            int kExt = kIndexExtended(
                  k,
                  loc->nx,
                  loc->ny,
                  loc->nf,
                  loc->halo.lt,
                  loc->halo.rt,
                  loc->halo.dn,
                  loc->halo.up);
            int x          = kxPos(k, loc->nx, loc->ny, loc->nf);
            int y          = kyPos(k, loc->nx, loc->ny, loc->nf);
            int f          = featureIndex(k, loc->nx, loc->ny, loc->nf);
            activity[kExt] = buffer.at(x, y, f);
         }
      }
   }
   return Response::SUCCESS;
}
