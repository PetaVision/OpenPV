/*
 * MLPOutputLayer.hpp
 * Author: slundquist
 */

#ifndef MLPOUTPUTLAYER_HPP_ 
#define MLPOUTPUTLAYER_HPP_ 
#include "MLPSigmoidLayer.hpp"

namespace PV{

class MLPOutputLayer : public PV::MLPSigmoidLayer{
public:
   MLPOutputLayer(const char * name, HyPerCol * hc);
   virtual ~MLPOutputLayer();
   virtual int updateState(double timef, double dt);
   int initialize(const char * name, HyPerCol * hc);
   int communicateInitInfo();
   virtual int allocateDataStructures();
protected:
   MLPOutputLayer();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_LocalTarget(enum ParamsIOFlag ioFlag);
   virtual void ioParam_StatProgressPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_GTLayername(enum ParamsIOFlag ioFlag);
private:
   bool localTarget;
   pvdata_t * classBuffer;
   int initialize_base();
   double statProgressPeriod;
   double nextStatProgress;
   HyPerLayer * gtLayer;
   char* gtLayername;
   //Variables for stats
   int numRight;
   int numWrong;
   int progressNumRight;
   int progressNumWrong;
};

}
#endif 
