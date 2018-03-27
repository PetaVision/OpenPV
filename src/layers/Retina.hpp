/*
 * Retina.h
 *
 *  Created on: Jul 29, 2008
 *
 */

#ifndef RETINA_HPP_
#define RETINA_HPP_

#include "HyPerLayer.hpp"
//#include "../kernels/Retina_params.h"
#include "../include/pv_types.h"
#include "../io/fileio.hpp"
#include "columns/Random.hpp"

#define NUM_RETINA_CHANNELS 2
#define NUM_RETINA_EVENTS 3
//#define EV_R_PHI_E    0
//#define EV_R_PHI_I    1
//#define EV_R_ACTIVITY 2

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
   friend int test_kernels(int argc, char *argv[]);

   Retina(const char *name, HyPerCol *hc);
   virtual ~Retina();
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;

   int setRetinaParams(PVParams *p);

   virtual Response::Status updateState(double time, double dt) override;

   virtual bool activityIsSpiking() override { return spikingFlag; }

  protected:
   Retina();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_spikingFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_foregroundRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_backgroundRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_beginStim(enum ParamsIOFlag ioFlag);
   virtual void ioParam_endStim(enum ParamsIOFlag ioFlag);
   virtual void ioParam_burstFreq(enum ParamsIOFlag ioFlag);
   virtual void ioParam_burstDuration(enum ParamsIOFlag ioFlag);
   virtual void ioParam_refractoryPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_absRefractoryPeriod(enum ParamsIOFlag ioFlag);
   virtual void allocateV() override;
   virtual Response::Status registerData(Checkpointer *checkpointer) override;
   virtual void initializeV() override;
   virtual void initializeActivity() override;
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;
   virtual void readRandStateFromCheckpoint(Checkpointer *checkpointer);

   bool spikingFlag; // specifies that layer is spiking
   Retina_params rParams; // used in update state
   Random *randState;
   float probStimParam;
   float probBaseParam;

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
