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

class CochlearLayer : public PV::ANNLayer{
public:
   CochlearLayer(const char* name, PV::HyPerCol * hc);
   virtual ~CochlearLayer();
   virtual int updateState (double time, double dt);
    
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   
   const std::vector <float> getTargetFreqs() {return targetFreqs;}
   const std::vector <float> getDampingConstants() {return dampingConstants;}
    const std::vector <float> getCochlearScales() {return cochlearScales;}
   float getSampleRate() { return sampleRate; }
    double getDisplayPeriod() {return displayPeriod; }
    double getnextDisplayTime() {return nextDisplayTime; }
    
protected:
   CochlearLayer();

   int initialize(const char * name, PV::HyPerCol * hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_FreqMinMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nf(enum ParamsIOFlag ioFlag);
   virtual void ioParam_targetChannel(enum ParamsIOFlag ioFlag);
   virtual void ioParam_inputLayername(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dampingConstant(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sampleRate(enum ParamsIOFlag ioFlag);
    virtual void ioParam_cochlearScale(enum ParamsIOFlag ioFlag);
    virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);
    
private:
   int initialize_base();
   float freqMin;
   float freqMax;
   std::vector <float> targetFreqs;
   std::vector <float> radianFreqs;
   std::vector <float> omegas;
   std::vector <float> dampingConstants;
    std::vector <float> cochlearScales;
   PV::HyPerLayer* inputLayer;
   char* inputLayername;
   int targetChannel;
   float dampingConstant;
    float omega;
   float sampleRate;
    float cochlearScale;
    double displayPeriod;
    double nextDisplayTime;
   float* vVal; //velocity value
   float* xVal; //x value
    float timestep;

}; // end of class CochlearLayer


PV::BaseObject * createCochlearLayer(char const * name, PV::HyPerCol * hc);

#endif /* COCHLEARLAYER_HPP_ */
