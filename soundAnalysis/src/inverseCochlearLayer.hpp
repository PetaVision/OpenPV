/*
 * inverseCochlearLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef INVERSECOCHLEARLAYER_HPP_
#define INVERSECOCHLEARLAYER_HPP_

#include <layers/ANNLayer.hpp>
#include "CochlearLayer.hpp"

namespace PV {

class inverseCochlearLayer : public ANNLayer{
public:
   inverseCochlearLayer(const char* name, HyPerCol * hc);
   virtual ~inverseCochlearLayer();
   virtual int updateState (double time, double dt);

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
protected:
   inverseCochlearLayer();

   int initialize(const char * name, HyPerCol * hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nf(enum ParamsIOFlag ioFlag);
   virtual void ioParam_inputLayername(enum ParamsIOFlag ioFlag);
   virtual void ioParam_cochlearLayername(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sampleRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_bufferLength(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();
   int ringBuffer(int level);
   
   
   float sampleRate;
   char* inputLayername;
   char* cochlearLayername;
   //The layer to grab the input from
   HyPerLayer* inputLayer;
   //The cochlear layer to grab nessessary parameters from
   CochlearLayer* cochlearLayer;
   
   int bufferLength;
   pvdata_t ** inputRingBuffer;
   int ringBufferLevel;
   double * timehistory;
   double * dthistory;
   
   float * targetFreqs;
   float * deltaFreqs;


}; // end of class inverseCochlearLayer

}  // end namespace PV

#endif /* ANNLAYER_HPP_ */
