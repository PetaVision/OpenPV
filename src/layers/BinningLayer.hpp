#ifndef BINNINGLAYER_HPP_
#define BINNINGLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class BinningLayer: public PV::HyPerLayer {
public:
   BinningLayer(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int initializeState();
   virtual int requireMarginWidth(int marginWidthNeeded, int * marginWidthResult);
   virtual ~BinningLayer();

protected:
   BinningLayer();
   int initialize(const char * name, HyPerCol * hc);
   int setParams(PVParams * params);
   void readOriginalLayerName(PVParams * params);
   void readBinMaxMin(PVParams * params);
   void readDelay(PVParams * params);
   void readBinSigma(PVParams * params);
   int allocateV();
   virtual int updateState(double timef, double dt);
   virtual int doUpdateState(double timed, double dt, const PVLayerLoc * origLoc,
         const PVLayerLoc * currLoc, const pvdata_t * origData, pvdata_t * currV, float binMax, float binMin);

   float getSigma(){return binSigma;}
   float calcNormDist(float xVal, float mean, float binSigma);
private:
   int initialize_base();
   int delay;
   float binMax;
   float binMin;
   float binSigma;

protected:
   char * originalLayerName;
   HyPerLayer * originalLayer;
};

} /* namespace PV */
#endif 
