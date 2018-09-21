/*
 * Retina.h
 *
 *  Created on: Jul 29, 2008
 *
 */

#ifndef RETINA_HPP_
#define RETINA_HPP_

#include "HyPerLayer.hpp"
#include "columns/Random.hpp"
#include "include/pv_types.h"
#include "io/fileio.hpp"

#define NUM_RETINA_CHANNELS 2 // excitatory and inhibitory

struct Retina_params {
   float probStim;
   float probBase;
   double beginStim;
   double endStim;
   float burstFreq; // frequency of bursts
   float burstDuration; // duration of each burst, <=0 -> sinusoidal

   float refractory_period;
   float abs_refractory_period;
};

namespace PV {

class Retina : public PV::HyPerLayer {
  public:
   // default refractory periods for neurons
   static constexpr float mDefaultAbsRefractoryPeriod = 3.0f;
   static constexpr float mDefaultRefractoryPeriod    = 5.0f;
   static const int mNumRetinaChannels                = 2; // excitatory and inhibitory

   Retina(const char *name, HyPerCol *hc);
   virtual ~Retina();

   int setRetinaParams(double deltaTime);

   virtual Response::Status updateState(double time, double dt) override;

   virtual bool activityIsSpiking() override { return spikingFlag; }

  protected:
   Retina();
   int initialize(const char *name, HyPerCol *hc);
   virtual InternalStateBuffer *createInternalState() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_spikingFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_foregroundRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_backgroundRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_beginStim(enum ParamsIOFlag ioFlag);
   virtual void ioParam_endStim(enum ParamsIOFlag ioFlag);
   virtual void ioParam_burstFreq(enum ParamsIOFlag ioFlag);
   virtual void ioParam_burstDuration(enum ParamsIOFlag ioFlag);
   virtual void ioParam_refractoryPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_absRefractoryPeriod(enum ParamsIOFlag ioFlag);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual Response::Status initializeState(std::shared_ptr<InitializeStateMessage const> message);
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;
   virtual void readRandStateFromCheckpoint(Checkpointer *checkpointer);

   bool spikingFlag; // specifies that layer is spiking
   Retina_params rParams; // used in update state
   Random *randState;
   float probStimParam;
   float probBaseParam;

   // An extended buffer used by spiking retinas. Holds for each neuron the elapsed time
   // since the last time that neuron spiked.
   float *mSinceLastSpike = nullptr;

  private:
   int initialize_base();

   // For input from a given source input layer, determine which
   // cells in this layer will respond to the input activity.
   // Return the feature vectors for both the input and the sensitive
   // neurons, since most likely we will have to determine those.
   int findPostSynaptic(
         int dim,
         int maxSize,
         int col,
         // input: which layer, which neuron
         HyPerLayer *lSource,
         float pos[],
         // output: how many of our neurons are connected.
         // an array with their indices.
         // an array with their feature vectors.
         int *nNeurons,
         int nConnectedNeurons[],
         float *vPos);

   int calculateWeights(HyPerLayer *lSource, float *pos, float *vPos, float *vfWeights);

}; // class Retina

} // namespace PV

#endif /* RETINA_HPP_ */
