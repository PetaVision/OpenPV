/*
 * RescaleActivityBuffer.cpp
 * Rescale layer is a cloneVLayer, grabs activity from orig layer and rescales it
 */

#ifndef RESCALEACTIVITYBUFFER_HPP_
#define RESCALEACTIVITYBUFFER_HPP_

#include "ActivityBuffer.hpp"

namespace PV {

class RescaleActivityBuffer : public ActivityBuffer {
  protected:
   void ioParam_targetMax(enum ParamsIOFlag ioFlag);
   void ioParam_targetMin(enum ParamsIOFlag ioFlag);
   void ioParam_targetMean(enum ParamsIOFlag ioFlag);
   void ioParam_targetStd(enum ParamsIOFlag ioFlag);

   /**
    * @brief rescaleMethod: can be one of
    *       maxmin, meanstd, pointmeanstd, pointResponseNormalization, softmax, l2, l2NoMean,
    * zerotonegative, logreg
    */
   void ioParam_rescaleMethod(enum ParamsIOFlag ioFlag);
   void ioParam_patchSize(enum ParamsIOFlag ioFlag);

  public:
   enum Method {
      UNDEFINED,
      MAXMIN,
      MEANSTD,
      POINTMEANSTD,
      POINTRESPONSENORMALIZATION,
      SOFTMAX,
      L2,
      L2NOMEAN,
      ZEROTONEGATIVE,
      LOGREG
   };

   RescaleActivityBuffer(const char *name, HyPerCol *hc);
   virtual ~RescaleActivityBuffer();
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   float getTargetMax() const { return mTargetMax; }
   float getTargetMin() const { return mTargetMin; }
   float getTargetMean() const { return mTargetMean; }
   float getTargetStd() const { return mTargetStd; }
   float getPatchSize() const { return mPatchSize; }
   const char *getRescaleMethod() const { return mRescaleMethod; }

   ActivityBuffer const *getOriginalBuffer() const { return mOriginalBuffer; }

  protected:
   RescaleActivityBuffer();
   int initialize(const char *name, HyPerCol *hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   float mTargetMax     = 1.0f;
   float mTargetMin     = -1.0f;
   float mTargetMean    = 0.0f;
   float mTargetStd     = 1.0f;
   char *mRescaleMethod = nullptr;
   Method mMethodCode   = UNDEFINED;
   int mPatchSize       = 1;

   ActivityBuffer *mOriginalBuffer = nullptr;
}; // class RescaleActivityBuffer

} // namespace PV

#endif /* CLONELAYER_HPP_ */
