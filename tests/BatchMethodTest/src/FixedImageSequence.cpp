#include "FixedImageSequence.hpp"
#include <components/ActivityBuffer.hpp>
#include <components/ActivityComponentActivityOnly.hpp>
#include <structures/Image.hpp>
#include <utils/BufferUtilsMPI.hpp>

FixedImageSequence::FixedImageSequence(char const *name, PV::HyPerCol *hc) {
   PV::HyPerLayer::initialize(name, hc);
}

PV::ActivityComponent *FixedImageSequence::createActivityComponent() {
   return new PV::ActivityComponentActivityOnly<PV::ActivityBuffer>(name, parent);
}

void FixedImageSequence::ioParam_triggerLayerName(enum PV::ParamsIOFlag ioFlag) {
   if (ioFlag == PV::PARAMS_IO_READ) {
      triggerLayerName = nullptr;
      triggerFlag      = false;
      parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", nullptr);
   }
}

PV::Response::Status
FixedImageSequence::initializeState(std::shared_ptr<PV::InitializeStateMessage const> message) {
   mActivityPointer =
         mActivityComponent->getComponentByType<PV::ActivityBuffer>()->getReadWritePointer();
   float *A = mActivityPointer;
   for (int k = 0; k < getNumNeuronsAllBatches(); k++) {
      A[k] = 0.0f;
   }
   defineImageSequence();
   return PV::Response::SUCCESS;
}

PV::Response::Status FixedImageSequence::updateState(double simTime, double deltaTime) {
   FatalIf(deltaTime != 1.0, "FixedImageSequence assumes dt = 1.\n");
   double timestampRounded = std::nearbyint(simTime);
   FatalIf(
         simTime != timestampRounded,
         "FixedImageSequence::updateState() requires the time argument be an integer.\n");
   PVLayerLoc const *loc = getLayerLoc();
   int timestampInt      = (int)timestampRounded;
   int globalBatchSize   = getMPIBlock()->getGlobalBatchDimension() * loc->nbatch;
   int localNBatch       = loc->nbatch;

   for (int m = 0; m < getMPIBlock()->getBatchDimension(); m++) {
      int mpiBlockIndex = m + getMPIBlock()->getStartBatch();
      for (int b = 0; b < localNBatch; b++) {
         PV::Buffer<float> buffer;
         if (getMPIBlock()->getRank() == 0) {
            int globalBatchElement = b + localNBatch * mpiBlockIndex;
            int inputIndex         = mIndexStart + (timestampInt - 1) * mIndexStepTime;
            inputIndex += mIndexStepBatch * globalBatchElement;
            inputIndex %= mNumImages;

            auto filename = std::string("input/images/") + std::to_string(inputIndex) + ".png";
            PV::Image image(filename);
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
         PV::BufferUtils::scatter<float>(getMPIBlock(), buffer, loc->nx, loc->ny, m, 0);
         if (getMPIBlock()->getBatchIndex() != m) {
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
   return PV::Response::SUCCESS;
}
