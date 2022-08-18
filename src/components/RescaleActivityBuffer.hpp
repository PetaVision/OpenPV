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
    *
    * maxmin:
    *    The region is linearly rescaled so that the min and max have values specified by the
    *    targetMin and targetMax parameters. That is, if m is the minimum of the original layer
    *    (across all (x,y) locations and feature dimension) and M is the maximum of the original
    *    layer the rescaled layer at a particular neuron has value
    *               (v-m)/(M-m)*(targetMax - targetMin) + targetMin
    *    where v is the value of the original layer at the corresponding neuron.
    *    If M and m are equal, i.e. the original layer is constant, the rescaled layer is
    *    set to zero everywhere.
    *
    * meanstd:
    *    The region is linearly rescaled so that the mean and standard deviation have the
    *    values specified by the targetMean and targetStd parameters. That is, if mu and sigma
    *    are the mean and standard deviation of the original layer (across all (x,y) locations and
    *    feature dimension), the rescaled layer at a particular neuron has value
    *               (v - mu)/std * targetStd + targetMean
    *    where v is the value of the original layer at the corresponding neuron.
    *    If M and m are equal, i.e. the original layer is constant, the rescaled layer is equal
    *    to the original layer.
    *
    * pointmeanstd:
    *    Similar to meanstd, except that the mean and std. dev. of the original layer is
    *    computed separately for each (x,y) location. That is, if mu_{ij} and sigma_{ij} are
    *    the mean and standard deviation across the feature dimension of the original layer
    *    at location x=x_i, y=y_j, the rescaled layer at x=x_i, y=y_j, f=f_k will be
    *               (v_{ijk} - mu_{ij}) / sigma_{ij} * targetStd + targetMean.
    *
    * pointResponseNormalization:
    *    At each (x,y) location, the values are rescaled by dividing by the standard deviation
    *    across the feature dimension. That is, if v_{ijk}, k=1,...,nf, are the values of the
    *    original layer at x=x_i, y=y_i, then the rescaled layer values are v_{ijk}/sigma_{ij}
    *    where sigma_{ij} is the standard deviation of v_{ij1}, v_{ij2}, ..., v_{ij(nf)}.
    *    If sigma_{ij} is zero, that is, if the values for all features at x=x_i, y=y_j are
    *    equal, the rescaled layer is equal to the original layer at that (x,y) location.
    * 
    * softmax:
    *    At each (x,y) location, the softmax function is applied to the values across features.
    *    That is, if v_{ijk} is the value of the original layer at x=x_i, y=y_j, f=f_k,
    *    then the value of rescaled layer at that neuron is exp(v_{ijk}/sum_p v_{ijp}).
    *
    * l2:
    *    The region is linearly rescaled to have mean zero and standard deviation
    *               sigma_rescaled = 1 / sqrt(patchSize)
    *    where sigma_original is the standard deviation of the original layer (across the x-, y-,
    *    and feature dimensions) and patchSize is the patchSize parameter.
    *    If the layer is constant (sigma_original == 0), the rescaled layer is the same as
    *    the original layer and a warning is sent to the error log.
    *
    * l2NoMean:
    *    The region is multiplied by a constant factor of
    *               1/(RMS_original * sqrt(patchSize)),
    *    where RMS_original is the root-mean-square of the original layer, across the
    *    x-, y-, and feature dimensions.
    *    If the layer is everywhere zero (M_original == 0), the rescaled layer is the same as
    *    the original layer and a warning is sent to the error log.
    *
    * zerotonegative:
    *    The rescaled layer copies the original layer, except that where the original layer
    *    has value zero, the rescaled layer has value -1.
    *
    * logreg:
    *    The rescaled layer is computed point-by-point from the original layer using the mapping
    *               v_rescaled = 1 / (1 + exp(v_original))
    *    that is, a logistic curve with limits of 1 at neg. infinity and 0 at pos. infinity.
    *
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

   RescaleActivityBuffer(const char *name, PVParams *params, Communicator const *comm);
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
   void initialize(const char *name, PVParams *params, Communicator const *comm);
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
