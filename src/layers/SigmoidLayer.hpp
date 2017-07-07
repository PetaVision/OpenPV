/*
 * SigmoidLayer.hpp
 * can be used to implement Sigmoid junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef SIGMOIDLAYER_HPP_
#define SIGMOIDLAYER_HPP_

#include "CloneVLayer.hpp"

namespace PV {

// SigmoidLayer can be used to implement Sigmoid junctions between spiking neurons
class SigmoidLayer : public CloneVLayer {
  public:
   SigmoidLayer(const char *name, HyPerCol *hc);
   virtual ~SigmoidLayer();
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;
   virtual int updateState(double timef, double dt) override;
   virtual int setActivity() override;

  protected:
   SigmoidLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_Vrest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VthRest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InverseFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_SigmoidFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_SigmoidAlpha(enum ParamsIOFlag ioFlag);
   /* static */ int updateState(
         double timef,
         double dt,
         const PVLayerLoc *loc,
         float *A,
         float *V,
         int num_channels,
         float *gSynHead,
         float Vth,
         float V0,
         float sigmoid_alpha,
         bool sigmoid_flag,
         bool inverse_flag);

  private:
   int initialize_base();
   float V0;
   float Vth;
   bool InverseFlag;
   bool SigmoidFlag;
   float SigmoidAlpha;
}; // class SigmoidLayer

} // namespace PV

#endif /* CLONELAYER_HPP_ */
