/*
 * AdaptiveTimestepProbe.hpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#ifndef ADAPTIVETIMESTEPPROBE_HPP_
#define ADAPTIVETIMESTEPPROBE_HPP_

#include "io/ColProbe.hpp"
#include "io/AdaptiveTimestepController.hpp"
#include "io/BaseProbe.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

// AdaptiveTimestepProbe to be a subclass of ColProbe since it doesn't belong
// to a layer or connection, and the HyPerCol has to know of its existence to
// call various methods. It doesn't use any ColProbe-specific behavior other
// than ColProbe inserting the probe into the HyPerCol's list of ColProbes.
// Once the observer pattern is more fully implemented, it could probably
// derive straight from BaseProbe.
class AdaptiveTimestepProbe: public ColProbe {
protected:
   /**
    * List of parameters needed from the AdaptiveTimestepProbe class
    * @name AdaptiveTimestepProbe Parameters
    * @{
    */

   /**
    * @brief targetName: the name of the probe that this probe attaches to.
    * The target probe's values are used as the input to compute the adaptive timesteps.
    */
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag);

   /**
    * @brief baseMax: If mDtAdaptController is set, specifies the maximum timescale allowed
    */
   virtual void ioParam_baseMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtScaleMin: If mDtAdaptController is set, specifies the default timescale
    * @details The parameter name is misleading, since dtAdapt can drop below timescale min
    */
   virtual void ioParam_dtScaleMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtChangeMax: If mDtAdaptController is set, specifies the upper limit of adaptive dt based on error
    * @details dt will only adapt if the percent change in error is between dtChangeMin and dtChangeMax
    */
   virtual void ioParam_dtChangeMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtChangeMin: If mDtAdaptController is set, specifies the lower limit of adaptive dt based on error
    * @details dt will only adapt if the percent change in error is between dtChangeMin and dtChangeMax.
    * Defaults to 0
    */
   virtual void ioParam_dtChangeMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief mDtMinToleratedTimeScale: If mDtAdaptController is set, specifies the minimum value dt can drop to before exiting
    * @details Program will exit if mTimeScale drops below this value
    */
   virtual void ioParam_dtMinToleratedTimeScale(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeTimescales: If mDtAdaptController is set, specifies if the timescales should be written
    * @details The timescales get written to outputPath/HyPerCol_timescales.txt.
    */
   virtual void ioParam_writeTimescales(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeTimeScaleFieldnames: A flag to determine if fieldnames are written to the HyPerCol_timescales file, if false, file is written as comma separated list
    */
   virtual void ioParam_writeTimeScaleFieldnames(enum ParamsIOFlag ioFlag);
   /** @} */

public:
   AdaptiveTimestepProbe(char const * name, HyPerCol * hc);
   virtual ~AdaptiveTimestepProbe();
   virtual int respond(std::shared_ptr<BaseMessage> const message) override;
   virtual int communicateInitInfo() override;
   virtual int allocateDataStructures() override;
   virtual int checkpointRead(const char * cpDir, double * timeptr) override;
   virtual int checkpointWrite(const char * cpDir) override;
   virtual int outputState(double timeValue) override;

protected:
   AdaptiveTimestepProbe();
   int initialize(char const * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   int respondAdaptTimestep(AdaptTimestepMessage const * message);
   bool needRecalc(double timeValue) override { return parent->simulationTime() > getLastUpdateTime(); }
   double referenceUpdateTime() const override { return mTargetProbe->getLastUpdateTime(); }
   int calcValues(double timeValue);

protected:
   double mBaseMax                  = 1.0;
   double mTimeScaleMin             = 1.0;
   double mDtMinToleratedTimeScale  = 1.0e-4;
   double mChangeTimeScaleMax       = 1.0;
   double mChangeTimeScaleMin       = 1.0;
   bool   mWriteTimescales          = true;
   bool   mWriteTimeScaleFieldnames = true;

   BaseProbe * mTargetProbe = nullptr;
   AdaptiveTimestepController * mAdaptiveTimestepController = nullptr;
};

} /* namespace PV */

#endif /* ADAPTIVETIMESTEPPROBE_HPP_ */
