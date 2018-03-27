#ifndef FIXEDIMAGESEQUENCE_HPP_
#define FIXEDIMAGESEQUENCE_HPP_

#include <layers/ImageLayer.hpp>

class FixedImageSequence : public PV::HyPerLayer {
  public:
   FixedImageSequence(char const *name, PV::HyPerCol *hc);
   virtual ~FixedImageSequence() {}

  protected:
   FixedImageSequence() {}

   /**
    * This layer type does not use the V buffer.
    */
   virtual void allocateV() override;

   /**
    * Initializes the activity buffer to zero and calls
    * the pure virtual method initIndices to set the
    * mIndexStart and mIndexSkip data members.
    */
   virtual PV::Response::Status initializeState() override;

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
   int const mNumImages = 10;
}; // end class FixedImageSequence

#endif // FIXEDIMAGESEQUENCE_HPP_
