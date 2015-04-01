/*
 * SoftMaxBackprop.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef SOFTMAXBACKPROPLAYER_HPP_
#define SOFTMAXBACKPROPLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class SoftMaxBackprop: public PV::ANNLayer {
public:
   SoftMaxBackprop(const char * name, HyPerCol * hc);
   virtual ~SoftMaxBackprop();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
protected:
/**
 * List of protect parameters for SoftMaxBackprop
 *
 * @{
 */
   SoftMaxBackprop();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int allocateV();
   virtual int initializeV();
   virtual int checkpointWrite(const char * cpDir);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ForwardLayername(enum ParamsIOFlag ioFlag);

   virtual int updateState(double time, double dt);
   virtual int readVFromCheckpoint(const char * cpDir, double * timeptr);
private:
   int initialize_base();
   char * forwardLayername;
   HyPerLayer * forwardLayer;
};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
