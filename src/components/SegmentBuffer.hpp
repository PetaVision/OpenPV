/*
 * SegmentBuffer.hpp
 *
 * created on: Jan 29, 2016
 *     Author: Sheng Lundquist
 */

#ifndef SEGMENTBUFFER_HPP_
#define SEGMENTBUFFER_HPP_

#include "components/ActivityBuffer.hpp"

#include "components/OriginalLayerNameParam.hpp"
#include <map>

namespace PV {

class SegmentBuffer : public ActivityBuffer {
  protected:
   void ioParam_segmentMethod(ParamsIOSwitch ioSwitch);

  public:
   SegmentBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~SegmentBuffer();

   virtual Response::Status allocateDataStructures() override;

   const std::map<int, int> getCenterIdxBuf(int batch) { return centerIdx[batch]; }

  protected:
   SegmentBuffer();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void setOriginalActivity(ObserverTable const *table);
   void checkDimensions();
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  private:
   int checkLabelBufSize(int newSize);
   int loadLabelBuf();
   int loadCenterIdxMap(int batchIdx, int numLabels);

   int checkIdxBufSize(int newSize);

  protected:
   int mLabelBufSize = 0;
   int *mLabelBuf    = nullptr;
   int *mMaxXBuf     = nullptr;
   int *mMaxYBuf     = nullptr;
   int *mMinXBuf     = nullptr;
   int *mMinYBuf     = nullptr;

   int centerIdxBufSize = 0;
   int *allLabelsBuf    = nullptr;
   int *centerIdxBuf    = nullptr;

   // Maps which go from segment label to max x/y global res index
   std::map<int, int> mMaxX;
   std::map<int, int> mMaxY;
   std::map<int, int> mMinX;
   std::map<int, int> mMinY;

   // Stores centriod linear index as global res
   std::vector<std::map<int, int>> centerIdx;

  private:
   // Data structures to keep track of segmentation labels and centroid idx
   ActivityBuffer *mOriginalActivity = nullptr;
   std::string segmentMethod;

}; // class SegmentBuffer

} /* namespace PV */
#endif
