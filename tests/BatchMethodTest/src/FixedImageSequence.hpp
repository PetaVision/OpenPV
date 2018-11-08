#ifndef FIXEDIMAGESEQUENCE_HPP_
#define FIXEDIMAGESEQUENCE_HPP_

#include <layers/HyPerLayer.hpp>

class FixedImageSequence : public PV::HyPerLayer {
  protected:
   /**
    * List of parameters needed from the HyPerLayer class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief triggerLayerName: FixedImageSequence always sets triggerLayerName to NULL
    */
   virtual void ioParam_triggerLayerName(enum PV::ParamsIOFlag ioFlag) override;
   /** @} */

  public:
   FixedImageSequence(char const *name, PV::PVParams *params, PV::Communicator *comm);
   virtual ~FixedImageSequence() {}

  protected:
   FixedImageSequence() {}

   PV::ActivityComponent *createActivityComponent() override;

   /**
    * Initializes the activity buffer to zero and calls
    * the pure virtual method defineImageSequence to set the
    * mIndexStart and mIndexSkip data members.
    */
   PV::Response::Status initializeState(std::shared_ptr<PV::InitializeStateMessage const> message);

   /**
    * A pure virtual method where derived classes should set the data members
    * mIndexStart, mIndexStepTime, and mIndexStepBatch.
    *
    * When time is t, global batch element b loads image number
    * (mIndexStart + (t-1) * mIndexStepTime + b) mod mNumImages.
    */
   virtual void defineImageSequence() = 0;

   /**
    * Uses timestamp and global batch index to load the image numbered
    * timestamp * globalBatchSize + globalBatchIndex
    * into the activity buffer.
    */
   virtual PV::Response::Status updateState(double timestamp, double dt) override;

   int getNumImages() const { return mNumImages; }

  protected:
   int mIndexStart;
   int mIndexStepTime;
   int mIndexStepBatch;
   float *mActivityPointer = nullptr;
   int const mNumImages    = 10;
}; // end class FixedImageSequence

#endif // FIXEDIMAGESEQUENCE_HPP_
