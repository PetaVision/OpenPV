/*
 * MLPOutputLayer.hpp
 * Author: slundquist
 */

#ifndef MLPOUTPUTLAYER_HPP_ 
#define MLPOUTPUTLAYER_HPP_ 
#include "MLPSigmoidLayer.hpp"

namespace PVMLearning{

class MLPOutputLayer : public MLPSigmoidLayer{
public:
   MLPOutputLayer(const char * name, PV::HyPerCol * hc);
   virtual ~MLPOutputLayer();
   virtual int updateState(double timef, double dt);
   int initialize(const char * name, PV::HyPerCol * hc);
   int communicateInitInfo();
   virtual int allocateDataStructures();
protected:
   MLPOutputLayer();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_LocalTarget(enum ParamsIOFlag ioFlag);
   virtual void ioParam_StatProgressPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_GTLayername(enum ParamsIOFlag ioFlag);
private:
   //TODO add more
   void multiclassNonlocalStats();
   void binaryNonlocalStats();
   void binaryLocalStats();

   bool localTarget;
   pvdata_t * classBuffer;
   int initialize_base();
   double statProgressPeriod;
   double nextStatProgress;
   PV::HyPerLayer * gtLayer;
   char* gtLayername;
   //Variables for stats
   int numRight;
   int numWrong;
   int progressNumRight;
   int progressNumWrong;
   //Variables for local stats
   int numTotPos;
   int numTotNeg;
   int truePos;
   int trueNeg;
   int progressNumTotPos;
   int progressNumTotNeg;
   int progressTruePos;
   int progressTrueNeg;
}; // end class MLPOutputLayer

PV::BaseObject * createMLPOutputLayer(char const * name, PV::HyPerCol * hc);

}  // end namespace PVMLearning
#endif 
