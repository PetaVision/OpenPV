/*
 * CochlearLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef COCHLEARLAYER_HPP_
#define COCHLEARLAYER_HPP_

#include <layers/ANNLayer.hpp>
#include <vector>

namespace PV {

class CochlearLayer : public ANNLayer{
public:
   CochlearLayer(const char* name, HyPerCol * hc);
   virtual ~CochlearLayer();
   virtual int updateState (double time, double dt);

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
protected:
   CochlearLayer();

   int initialize(const char * name, HyPerCol * hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_FreqMinMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nf(enum ParamsIOFlag ioFlag);
   virtual void ioParam_targetChannel(enum ParamsIOFlag ioFlag);
   virtual void ioParam_inputLayername(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dampingConstant(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sampleRate(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();
   float freqMin;
   float freqMax;
   std::vector <float> targetFreqs;
   std::vector <float> radianFreqs;
   std::vector <float> omegas;
   std::vector <float> dampingConstants;
   HyPerLayer* inputLayer;
   char* inputLayername;
   int targetChannel;
   float dampingConstant;
    float omega;
   float sampleRate;
   float* vVal; //velocity value
   float* xVal; //x value
    float timestep;

}; // end of class CochlearLayer

}  // end namespace PV

#endif /* ANNLAYER_HPP_ */
