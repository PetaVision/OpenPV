#ifndef SEGMENTIFY_HPP_
#define SEGMENTIFY_HPP_

#include "HyPerLayer.hpp"
#include "SegmentLayer.hpp"

namespace PV {

class Segmentify: public PV::HyPerLayer {
public:
   Segmentify(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual bool activityIsSpiking() { return false; }
   virtual ~Segmentify();

protected:
   Segmentify();
   int initialize(const char * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   void ioParam_segmentLayerName(enum ParamsIOFlag ioFlag);
   //Defines the way to reduce values within a segment
   //into a single scalar. Options are "average", "sum", and "max".
   void ioParam_inputMethod(enum ParamsIOFlag ioFlag);
   //Defines the way to fill the output segment with the
   //reduced scalar method. Options are "centroid" and "fill"
   void ioParam_outputMethod(enum ParamsIOFlag ioFlag);
   int allocateV();
   int initializeV();
   virtual int initializeActivity();

   virtual int updateState(double timef, double dt);

   float calcNormDist(float xVal, float mean, float binSigma);
private:
   int initialize_base();

protected:
   int checkLabelValBuf(int newSize);
   int buildLabelToIdx(int batchIdx);
   int calculateLabelVals(int batchIdx);
   int setOutputVals(int batchIdx);

   char * originalLayerName;
   HyPerLayer * originalLayer;
   char * segmentLayerName;
   SegmentLayer * segmentLayer;

   //Reusing this buffer for batches
   //Map to go from label to index into labelVals
   std::map<int, int> labelToIdx;
   //Matrix to store values (one dim for features, one for # labels
   int numLabelVals;
   int* labelIdxBuf;
   float** labelVals;
   int** labelCount;

   char* inputMethod;
   char* outputMethod;

}; // class Segmentify

BaseObject * createSegmentify(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif 
