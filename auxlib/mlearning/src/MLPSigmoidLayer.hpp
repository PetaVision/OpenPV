/*
 * MLPSigmoidLayer.hpp
 * can be used to implement Sigmoid junctions
 *
 */

#ifndef MLPSIGMOIDLAYER_HPP_
#define MLPSIGMOIDLAYER_HPP_

#include <layers/CloneVLayer.hpp>
#include "MLPForwardLayer.hpp"

namespace PVMLearning {

// MLPSigmoidLayer can be used to implement Sigmoid junctions between spiking neurons
class MLPSigmoidLayer: public PV::CloneVLayer {
public:
   MLPSigmoidLayer(const char * name, PV::HyPerCol * hc);
   virtual ~MLPSigmoidLayer();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
   virtual int setActivity();
protected:
   MLPSigmoidLayer();
   int initialize(const char * name, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_SymSigmoid(enum ParamsIOFlag ioFlag);

   virtual void ioParam_LinAlpha(enum ParamsIOFlag ioFlag);

   virtual void ioParam_Vrest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VthRest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InverseFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_SigmoidFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_SigmoidAlpha(enum ParamsIOFlag ioFlag);

   int updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, float Vth, float V0, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag, float linear_alpha, bool* dropout_buf);

private:
   int initialize_base();
   float linAlpha;
   bool* dropout;
   bool symSigmoid;

   float V0;
   float Vth;
   bool  InverseFlag;
   bool  SigmoidFlag;
   float SigmoidAlpha;
}; // end class MLPSigmoidLayer

PV::BaseObject * createMLPSigmoidLayer(char const * name, PV::HyPerCol * hc);

}  // end namespace PVMLearning

#endif /* CLONELAYER_HPP_ */
