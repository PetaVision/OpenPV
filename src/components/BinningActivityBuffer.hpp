/*
 * BinningActivityBuffer.hpp
 *
 *  Created on: Jan 15, 2014
 *      Author: Sheng Lundquist
 */

#ifndef BINNINGACTIVITYBUFFER_HPP_
#define BINNINGACTIVITYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"
#include "components/ComponentBuffer.hpp"
#include "components/InternalStateBuffer.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

/**
 * A component for the activity updater for BinningLayer.
 */
class BinningActivityBuffer : public ActivityBuffer {
  protected:
   /**
    * List of parameters used by the BinningActivityBuffer class
    * @name ANNLayer Parameters
    * @{
    */

   void ioParam_binMin(enum ParamsIOFlag ioFlag);
   void ioParam_binMax(enum ParamsIOFlag ioFlag);
   void ioParam_delay(enum ParamsIOFlag ioFlag);
   void ioParam_binSigma(enum ParamsIOFlag ioFlag);
   void ioParam_zeroNeg(enum ParamsIOFlag ioFlag);
   void ioParam_zeroDCR(enum ParamsIOFlag ioFlag);
   void ioParam_normalDist(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   BinningActivityBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~BinningActivityBuffer();

   float getBinSigma() const { return mBinSigma; }

  protected:
   BinningActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * BinningActivityBuffer does not have an InternalStateBuffer.
    * However, the original layer must have the same nx and ny as the current layer,
    * and the original layer must have exactly one feature.
    */
   void checkDimensions() const;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   float calcGaussian(float x, float sigma);

  protected:
   float mBinMin    = 0.0f;
   float mBinMax    = 1.0f;
   int mDelay       = 0;
   float mBinSigma  = 0;
   bool mZeroNeg    = true;
   bool mZeroDCR    = false;
   bool mNormalDist = true;

   HyPerLayer *mOriginalLayer = nullptr;
};

} // namespace PV

#endif // BINNINGACTIVITYBUFFER_HPP_
