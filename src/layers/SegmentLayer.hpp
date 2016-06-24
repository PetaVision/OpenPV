#ifndef SEGMENTLAYER_HPP_
#define SEGMENTLAYER_HPP_

#include "HyPerLayer.hpp"
#include <map>

namespace PV {

class SegmentLayer: public PV::HyPerLayer {
public:
   SegmentLayer(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual bool activityIsSpiking() { return false; }
   virtual ~SegmentLayer();
   const std::map<int, int> getCenterIdxBuf(int batch){ return centerIdx[batch];}

protected:
   SegmentLayer();
   int initialize(const char * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   void ioParam_segmentMethod(enum ParamsIOFlag ioFlag);
   virtual int initializeActivity();

   virtual int allocateV();
   virtual int initializeV();

   virtual int updateState(double timef, double dt);

private:
   int initialize_base();
   int checkLabelBufSize(int newSize);
   int loadLabelBuf();
   int loadCenterIdxMap(int batchIdx, int numLabels);

   int checkIdxBufSize(int newSize);

   //Data structures to keep track of segmentation labels and centroid idx
   char * segmentMethod;
   char * originalLayerName;
   HyPerLayer* originalLayer;

protected:

   int labelBufSize;
   int* labelBuf;
   int* maxXBuf;
   int* maxYBuf;
   int* minXBuf;
   int* minYBuf;

   int centerIdxBufSize;
   int* allLabelsBuf;
   int* centerIdxBuf;
   


   //Maps which go from segment label to max x/y global res index
   std::map<int, int> maxX;
   std::map<int, int> maxY;
   std::map<int, int> minX;
   std::map<int, int> minY;
   
   //Stores centriod linear index as global res
   std::vector<std::map<int, int> > centerIdx;
   
}; // class SegmentLayer

BaseObject * createSegmentLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif 
