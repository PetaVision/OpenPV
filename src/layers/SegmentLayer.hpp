#ifndef SEGMENTLAYER_HPP_
#define SEGMENTLAYER_HPP_

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include <map>

namespace PV {

class SegmentLayer : public PV::HyPerLayer {
  public:
   SegmentLayer(const char *name, HyPerCol *hc);
   virtual Response::Status allocateDataStructures() override;
   virtual bool activityIsSpiking() override { return false; }
   virtual ~SegmentLayer();
   const std::map<int, int> getCenterIdxBuf(int batch) { return centerIdx[batch]; }

  protected:
   SegmentLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual void setObserverTable() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual LayerInputBuffer *createLayerInput();
   virtual InternalStateBuffer *createInternalState();
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void ioParam_segmentMethod(enum ParamsIOFlag ioFlag);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void setOriginalLayer();
   virtual void initializeActivity() override;

   virtual Response::Status updateState(double timef, double dt) override;

  private:
   int initialize_base();
   int checkLabelBufSize(int newSize);
   int loadLabelBuf();
   int loadCenterIdxMap(int batchIdx, int numLabels);

   int checkIdxBufSize(int newSize);

   // Data structures to keep track of segmentation labels and centroid idx
   char *segmentMethod;
   HyPerLayer *mOriginalLayer = nullptr;

  protected:
   int labelBufSize;
   int *labelBuf;
   int *maxXBuf;
   int *maxYBuf;
   int *minXBuf;
   int *minYBuf;

   int centerIdxBufSize;
   int *allLabelsBuf;
   int *centerIdxBuf;

   // Maps which go from segment label to max x/y global res index
   std::map<int, int> maxX;
   std::map<int, int> maxY;
   std::map<int, int> minX;
   std::map<int, int> minY;

   // Stores centriod linear index as global res
   std::vector<std::map<int, int>> centerIdx;

}; // class SegmentLayer

} /* namespace PV */
#endif
