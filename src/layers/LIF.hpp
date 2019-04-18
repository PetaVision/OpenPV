/*
 * LIF.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 *
 */

#ifndef LIF_HPP_
#define LIF_HPP_

#include "../columns/Random.hpp"
#include "HyPerLayer.hpp"
//#include "../kernels/LIF_params.h"

#define NUM_LIF_EVENTS 4
//#define EV_LIF_GSYN_E     0
//#define EV_LIF_GSYN_I     1
#define EV_LIF_GSYN_IB 2
//#define EV_LIF_ACTIVITY  3
#define pvconductance_t float

struct LIF_params {
   float Vrest;
   float Vexc;
   float Vinh;
   float VinhB;

   float tau;
   float tauE;
   float tauI;
   float tauIB;

   float VthRest;
   float tauVth;
   float deltaVth;
   float deltaGIB;

   float noiseFreqE;
   float noiseAmpE;
   float noiseFreqI;
   float noiseAmpI;
   float noiseFreqIB;
   float noiseAmpIB;
};

namespace PV {

class LIF : public PV::HyPerLayer {
  public:
   friend int test_kernels(int argc, char *argv[]);
   friend int test_LIF(int argc, char *argv[]);

   LIF(const char *name, HyPerCol *hc);
   LIF(const char *name, HyPerCol *hc, int num_channels);
   virtual ~LIF();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status registerData(Checkpointer *checkpointer) override;

   virtual Response::Status updateState(double time, double dt) override;
   virtual int setActivity() override;

   float *getVth() { return Vth; }
   virtual pvconductance_t *getConductance(ChannelType ch) {
      return ch < this->numChannels ? G_E + ch * getNumNeurons() : NULL;
   }

   virtual float getChannelTimeConst(enum ChannelType channel_type) override;

   virtual LIF_params *getLIFParams() { return &lParams; };

   virtual bool activityIsSpiking() override { return true; }

  protected:
   LIF_params lParams;
   Random *randState;
   float *Vth; // threshold potential
   pvconductance_t *G_E; // excitatory conductance
   pvconductance_t *G_I; // inhibitory conductance
   pvconductance_t *G_IB;

   char *methodString; // 'arma', 'before', or 'original'
   char method; // 'a', 'b', or 'o', the first character of methodString

  protected:
   LIF();
   int initialize(const char *name, HyPerCol *hc, const char *kernel_name);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_Vrest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Vexc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Vinh(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VinhB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VthRest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tau(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauE(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauI(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauIB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauVth(enum ParamsIOFlag ioFlag);
   virtual void ioParam_deltaVth(enum ParamsIOFlag ioFlag);
   virtual void ioParam_deltaGIB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseAmpE(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseAmpI(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseAmpIB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseFreqE(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseFreqI(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseFreqIB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_method(enum ParamsIOFlag ioFlag);
   virtual void allocateBuffers() override;
   virtual void allocateConductances(int num_channels);
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;
   virtual void readVthFromCheckpoint(Checkpointer *checkpointer);
   virtual void readG_EFromCheckpoint(Checkpointer *checkpointer);
   virtual void readG_IFromCheckpoint(Checkpointer *checkpointer);
   virtual void readG_IBFromCheckpoint(Checkpointer *checkpointer);
   virtual void readRandStateFromCheckpoint(Checkpointer *checkpointer);

  private:
   int initialize_base();
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
}; // class LIF

} // namespace PV

#endif /* LIF_HPP_ */
