#ifndef BINNINGLAYER_HPP_
#define BINNINGLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

/**
 * A layer class to sort another layer's output activity into bins.
 * The number of features of the BinningLayer is the number of bins.
 * The number of features of the other layer must be equal to 1.
 * In the simplest case, with binSigma==0, the region [binMin, binMax] is divided into nf equal
 * intervals, labeled 0, 1, ..., nf-1.
 * If the input activity at the location (x,y) falls into the bin labeled k, then the BinningLayer
 * has A(x,y,k) = 1 and A(x,y,k') = 0 if k != k'. If any input activity is less than the binMin
 * parameter, it is put in the bin labeled 0; similarly, input activity greater than binMax is put
 * in the bin labeled nf-1.
 *
 * Other parameters can modify the behavior of the BinningLayer, as described
 * in the documentation for those parameters.
 */
class BinningLayer : public PV::HyPerLayer {

  protected:
   /**
    * List of parameters needed from the BinningLayer class
    * @name BinningLayer Parameters
    * @{
    */

   /**
    * The layer whose activity values should be binned. The number of features of the
    * original layer must be 1.
    */
   void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);

   /**
    * The expected minimum value of input activity. Input activity below binMin is treated as if
    * it were equal to binMin.
    */
   void ioParam_binMin(enum ParamsIOFlag ioFlag);

   /**
    * The expected maximum value of input activity. Input activity above binMax is treated as if
    * it were equal to binMax.
    */
   void ioParam_binMax(enum ParamsIOFlag ioFlag);

   /**
    * Allows the binning to be performed on the input activity from the specified number of
    * timesteps in the past. Note that delay is given in timesteps, unlike connections where
    * it is specified in units of time.
    */
   void ioParam_delay(enum ParamsIOFlag ioFlag);

   /**
    * If binSigma is set to zero, BinningLayer performs one-hot binning as given in the
    * documentation.
    * If binSigma is positive, the value for each feature is given by a Gaussian curve whose
    * standard deviation is binSigma multiplied by the bin width ((binMax-binMin)/nf.
    * The Gaussian is normalized to have a maximum value of one.
    * Note that this maximum value only occurs if the input activity falls exactly at a bin center.
    */
   void ioParam_binSigma(enum ParamsIOFlag ioFlag);

   /**
    * If this flag is set, the non-active bins take the value zero.
    * If the flag is not set, non-active bins take the value -1.
    */
   void ioParam_zeroNeg(enum ParamsIOFlag ioFlag);

   /**
    * If this flag is set, an input activity of zero is interpreted as a don't-care region:
    * all features for that input activity's location are given the value specified by zeroNeg.
    * If the flag is false, an input activity of zero is treated like an ordinary value.
    */
   void ioParam_zeroDCR(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   BinningLayer(const char *name, HyPerCol *hc);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;
   virtual bool activityIsSpiking() override { return false; }
   virtual ~BinningLayer();
   float getBinSigma() { return mBinSigma; }

  protected:
   BinningLayer();
   int initialize(const char *name, HyPerCol *hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void allocateV() override;
   virtual void initializeV() override;
   virtual void initializeActivity() override;
   virtual Response::Status updateState(double simTime, double dt) override;

   float calcGaussian(float x, float sigma);

  private:
   int initialize_base();
   int mDelay      = 0;
   float mBinMax   = 0.0f;
   float mBinMin   = 1.0f;
   float mBinSigma = 0.0f;
   bool mZeroNeg   = true;
   bool mZeroDCR   = false;

  protected:
   char *mOriginalLayerName   = nullptr;
   HyPerLayer *mOriginalLayer = nullptr;
}; // class BinningLayer

} /* namespace PV */
#endif
