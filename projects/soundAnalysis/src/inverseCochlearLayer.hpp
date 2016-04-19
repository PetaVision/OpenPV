/*
 * inverseCochlearLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef INVERSECOCHLEARLAYER_HPP_
#define INVERSECOCHLEARLAYER_HPP_

#include <layers/ANNLayer.hpp>
#include <NewCochlear.h>

class inverseCochlearLayer : public PV::ANNLayer{
public:
   inverseCochlearLayer(const char* name, PV::HyPerCol * hc);
   virtual ~inverseCochlearLayer();
   virtual int updateState (double time, double dt);

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
protected:
   inverseCochlearLayer();

   int initialize(const char * name, PV::HyPerCol * hc);

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
   PV::HyPerLayer* inputLayer;
   //The cochlear layer to grab nessessary parameters from
   PVsound::NewCochlearLayer* cochlearLayer;
   
   int bufferLength;
   pvdata_t ** xhistory; // ring buffer of past x_k(t_j).
   int ringBufferLevel;
   double * timehistory; // may not need
   
   int numFrequencies;
   float * targetFreqs;
   float * deltaFreqs;
    float * radianFreqs;
   float ** Mreal; // f = sum_j M[j][k] * x_k(t_j).  Should choose a more descriptive name
   float ** Mimag;
   double nextDisplayTime;


}; // end of class inverseCochlearLayer

PV::BaseObject * create_inverseCochlearLayer(char const * name, PV::HyPerCol * hc);

#endif /* INVERSECOCHLEARLAYER_HPP_ */
