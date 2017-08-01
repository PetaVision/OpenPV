/*
 * AdaptiveTimeScaleProbe.hpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#ifndef ADAPTIVETIMESCALEPROBE_HPP_
#define ADAPTIVETIMESCALEPROBE_HPP_

#include "BaseProbe.hpp"
#include "ColProbe.hpp"
#include "components/AdaptiveTimeScaleController.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

// AdaptiveTimeScaleProbe to be a subclass of ColProbe since it doesn't belong
// to a layer or connection, and the HyPerCol has to know of its existence to
// call various methods. It doesn't use any ColProbe-specific behavior other
// than ColProbe inserting the probe into the HyPerCol's list of ColProbes.
// Once the observer pattern is more fully implemented, it could probably
// derive straight from BaseProbe.
class AdaptiveTimeScaleProbe : public ColProbe {
  protected:
   /**
    * List of parameters needed from the AdaptiveTimeScaleProbe class
    * @name AdaptiveTimeScaleProbe Parameters
    * @{
    */

   /**
    * @brief targetName: the name of the probe that this probe attaches to.
    * The target probe's values are used as the input to compute the adaptive
    * timesteps.
    */
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief baseMax: Specifies the initial maximum timescale allowed.
    * The maximum timescale is allowed to increase at a rate specified
    * by the growthFactor parameter.
    */
   virtual void ioParam_baseMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief baseMin: Specifies the minimum timescale allowed.
    */
   virtual void ioParam_baseMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief tauFactor: If Specifies the coefficient on the effective decay rate
    * used to compute
    * the timescale.
    */
   virtual void ioParam_tauFactor(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtChangeMin: Specifies the percentage by which the maximum timescale
    * increases
    * when the timescale reaches the maximum.
    */
   virtual void ioParam_growthFactor(enum ParamsIOFlag ioFlag);

   // writeTimeScales was marked obsolete Jul 27, 2017.
   /**
    * @brief writeTimeScales is obsolete, as it is redundant with textOutputFlag.
    */
   virtual void ioParam_writeTimeScales(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeTimeScaleFieldnames: A flag to determine if fieldnames are
    * written to the
    * HyPerCol_timescales file. If false, file is written as comma separated list
    */
   virtual void ioParam_writeTimeScaleFieldnames(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   AdaptiveTimeScaleProbe(char const *name, HyPerCol *hc);
   virtual ~AdaptiveTimeScaleProbe();
   virtual int respond(std::shared_ptr<BaseMessage const> message) override;
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;
   virtual int outputState(double timeValue) override;

  protected:
   AdaptiveTimeScaleProbe();
   int initialize(char const *name, HyPerCol *hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual int registerData(Checkpointer *checkpointer) override;
   int respondAdaptTimestep(std::shared_ptr<AdaptTimestepMessage const> message);
   bool needRecalc(double timeValue) override {
      return parent->simulationTime() > getLastUpdateTime();
   }
   double referenceUpdateTime() const override { return parent->simulationTime(); }
   int calcValues(double timeValue) override;
   virtual bool needUpdate(double timeValue, double dt) override { return true; }
   virtual void allocateTimeScaleController();

  protected:
   double mBaseMax                = 1.0;
   double mBaseMin                = 1.0;
   double tauFactor               = 1.0;
   double mGrowthFactor           = 1.0;
   bool mWriteTimeScaleFieldnames = true;

   BaseProbe *mTargetProbe                                   = nullptr;
   AdaptiveTimeScaleController *mAdaptiveTimeScaleController = nullptr;
};

} /* namespace PV */

#endif /* ADAPTIVETIMESCALEPROBE_HPP_ */
