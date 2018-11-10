/*
 * RetinaActivityBuffer.hpp
 *
 *  Created on: Jul 29, 2008
 */

#ifndef RETINAACTIVITYBUFFER_HPP_
#define RETINAACTIVITYBUFFER_HPP_

#include "columns/Random.hpp"
#include "components/ActivityBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

struct RetinaParams {
   float mProbStim;
   float mProbBase;
   double mBeginStim;
   double mEndStim;
   float mBurstFreq; // frequency of bursts
   float mBurstDuration; // duration of each burst, <=0 -> sinusoidal

   float mRefractoryPeriod;
   float mAbsRefractoryPeriod;
};

/**
 * RetinaActivityBuffer is the ActivityBuffer subclass for the Retina layer.
 */
class RetinaActivityBuffer : public ActivityBuffer {
  protected:
   /**
    * List of parameters used by the RetinaActivityBuffer class
    * @name Retina Parameters
    * @{
    */

   /**
    * @brief spikingFlag:
    * If true, the retina produces a spike train whose rates depend on the
    * input. If false, the retina treats the input like a HyPerLayer.
    */
   virtual void ioParam_spikingFlag(enum ParamsIOFlag ioFlag);

   /**
    * The firing rate when the input is zero.
    */
   virtual void ioParam_backgroundRate(enum ParamsIOFlag ioFlag);

   /**
    * The amount by which the firing rate increases as the input
    * increases by one unit.
    */
   virtual void ioParam_foregroundRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_beginStim(enum ParamsIOFlag ioFlag);
   virtual void ioParam_endStim(enum ParamsIOFlag ioFlag);
   virtual void ioParam_burstFreq(enum ParamsIOFlag ioFlag);
   virtual void ioParam_burstDuration(enum ParamsIOFlag ioFlag);
   virtual void ioParam_refractoryPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_absRefractoryPeriod(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   // default refractory periods for neurons
   static constexpr float mDefaultAbsRefractoryPeriod = 3.0f;
   static constexpr float mDefaultRefractoryPeriod    = 5.0f;

   RetinaActivityBuffer(char const *name, HyPerCol *hc);

   virtual ~RetinaActivityBuffer();

  protected:
   RetinaActivityBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   // TODO: Eliminate code duplication with LIF - Make RandState a component
   void registerRandState(Checkpointer *checkpointer);

   void registerSinceLastSpike(Checkpointer *checkpointer);

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   /**
    * Converts the foreground rate and background rate parameters, given in hertz,
    * to probabilities of firing.
    */
   void setRetinaParams(double deltaTime);

   // TODO: Eliminate code duplication with LIF - Make RandState a component
   Response::Status readStateFromCheckpoint(Checkpointer *checkpointer);

   void readRandStateFromCheckpoint(Checkpointer *checkpointer);

   void readSinceLastSpikeFromCheckpoint(Checkpointer *checkpointer);

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   static float calcBurstStatus(double timed, RetinaParams *retinaParams);

   static int
   spike(float simTime,
         float deltaTime,
         float timeSinceLast,
         float stimFactor,
         taus_uint4 *rnd_state,
         float burstStatus,
         RetinaParams *retinaParams);

   static void spikingUpdateBuffer(
         const int nbatch,
         const int numNeurons,
         const double simTime,
         const double deltaTime,
         const int nx,
         const int ny,
         const int nf,
         const int lt,
         const int rt,
         const int dn,
         const int up,
         RetinaParams *retinaParams,
         taus_uint4 *rnd,
         float const *GSynHead,
         float *activity,
         float *timeSinceLast);

   static void nonspikingUpdateBuffer(
         const int nbatch,
         const int numNeurons,
         const double simTime,
         const double deltaTime,
         const int nx,
         const int ny,
         const int nf,
         const int lt,
         const int rt,
         const int dn,
         const int up,
         RetinaParams *retinaParams,
         float const *GSynHead,
         float *activity);

  protected:
   bool mSpikingFlag      = true;
   double mBackgroundRate = 0.0;
   double mForegroundRate = 1.0;
   double mBeginStim      = 0.0;
   double mEndStim        = FLT_MAX;
   float mBurstFreq       = 1.0f; // frequency of bursts
   float mBurstDuration   = 1000.0f; // duration of each burst, <=0 -> sinusoidal
   float mProbBase;
   float mProbStim;

   float mRefractoryPeriod;
   float mAbsRefractoryPeriod;

   RetinaParams mRetinaParams; // used in update state

   LayerInputBuffer *mLayerInput = nullptr;

   std::vector<float> mSinceLastSpike;
   Random *mRandState = nullptr;
};

} // namespace PV

#endif // RETINAACTIVITYBUFFER_HPP_
