/*
 * MLPErrorLayer.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef MLPERRORLAYER_HPP_
#define MLPERRORLAYER_HPP_

#include "ANNLayer.hpp"
#include "MLPForwardLayer.hpp"

namespace PV {

class MLPErrorLayer: public PV::ANNLayer {
public:
   MLPErrorLayer(const char * name, HyPerCol * hc);
   virtual ~MLPErrorLayer();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
protected:
   MLPErrorLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int allocateV();
   virtual int initializeV();
   virtual int checkpointWrite(const char * cpDir);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ForwardLayername(enum ParamsIOFlag ioFlag);
   virtual void ioParam_LossFunction(enum ParamsIOFlag ioFlag);
   virtual void ioParam_lastError(enum ParamsIOFlag ioFlag);

   virtual void ioParam_symSigmoid(enum ParamsIOFlag ioFlag);

   virtual void ioParam_Vrest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VthRest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_SigmoidAlpha(enum ParamsIOFlag ioFlag);

   virtual void ioParam_LinAlpha(enum ParamsIOFlag ioFlag);

   virtual int updateState(double time, double dt);
   virtual int readVFromCheckpoint(const char * cpDir, double * timeptr);
private:
   int initialize_base();
   bool * dropout;
   float Vrest;
   float VthRest;
   float sigmoid_alpha;
   char * forwardLayername;
   char * lossFunction;
   MLPForwardLayer* forwardLayer;
   float linAlpha;
   bool symSigmoid;
   bool lastError;
};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
