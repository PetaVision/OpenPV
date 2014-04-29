/*
 * MLPForwardLayer.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef MLPFORWARDLAYER_HPP_
#define MLPFORWARDLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class MLPForwardLayer: public PV::ANNLayer {
public:
   MLPForwardLayer(const char * name, HyPerCol * hc);
   virtual ~MLPForwardLayer();
   float * getBias() {return bias;}
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int checkpointWrite(const char * cpDir);
protected:
   MLPForwardLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InitBiasType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_BiasFilename(enum ParamsIOFlag ioFlag);
   virtual void ioParam_DropoutChance(enum ParamsIOFlag ioFlag);
   virtual void ioParam_NormFactor(enum ParamsIOFlag ioFlag);
   virtual int updateState(double time, double dt);
private:
   int readBias(const char * filename);
   int initialize_base();
   float * bias;
   char * initBiasType;
   char * biasFilename;
   float dropoutChance;
   Random * randState;
   float normFactor;
};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
