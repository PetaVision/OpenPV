/*
 * LCALIFLayer.hpp
 *
 *  Created on: Oct 3, 2012
 *      Author: slundquist
 */

#ifndef LCALIFLAYER_HPP_
#define LCALIFLAYER_HPP_

#include "HyPerLayer.hpp"
#include "LIFGap.hpp"

#define DEFAULT_DYNVTHSCALE 1.0f

namespace PV {
class LCALIFLayer : public PV::LIFGap {
  public:
   LCALIFLayer(const char *name, HyPerCol *hc); // The constructor called by other methods
   virtual ~LCALIFLayer();
   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status updateState(double timef, double dt) override;
   int findFlag(int numMatrixCol, int numMatrixRow);

   inline float getTargetRate() { return targetRateHz; }
   const float *getVadpt() { return Vadpt; }
   const float *getIntegratedSpikeCount() { return integratedSpikeCount; }
   const float *getVattained() { return Vattained; }
   const float *getVmeminf() { return Vmeminf; }

  protected:
   LCALIFLayer();
   int initialize(const char *name, HyPerCol *hc, const char *kernel_name);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_tauTHR(enum ParamsIOFlag ioFlag);
   virtual void ioParam_targetRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeInput(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Vscale(enum ParamsIOFlag ioFlag);
   virtual Response::Status registerData(Checkpointer *checkpointer) override;
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;
   virtual void read_integratedSpikeCountFromCheckpoint(Checkpointer *checkpointer);
   virtual void readVadptFromCheckpoint(Checkpointer *checkpointer);

   virtual void allocateBuffers() override;

   float *integratedSpikeCount; // plasticity decrement variable for postsynaptic layer
   float *G_Norm; // Copy of GSyn[CHANNEL_NORM] to be written out during checkpointing
   float *GSynExcEffective; // What is used as GSynExc, after normalizing, stored for checkpointing
   float *GSynInhEffective; // What is used as GSynInh
   float *excitatoryNoise;
   float *inhibitoryNoise;
   float *inhibNoiseB;
   float tauTHR;
   float targetRateHz;
   float Vscale;
   float *Vadpt;
   float *Vattained; // Membrane potential before testing to see if a spike resets it to resting
   // potential.  Output in checkpoints for diagnostic purposes but not otherwise
   // used.
   float *Vmeminf; // Asymptotic value of the membrane potential.  Output in checkpoints for
   // diagnostic purposes but not otherwise used.
   bool normalizeInputFlag;
   // other methods and member variables
  private:
   int initialize_base();
   // other methods and member variables
}; // class LCALIFLayer

} // namespace PV

#endif /* LCALIFLAYER_HPP_ */
