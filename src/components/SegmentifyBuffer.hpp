/*
 * SegmentifyBuffer.hpp
 *
 * created on: Feb 10, 2016
 *     Author: Sheng Lundquist
 */

#ifndef SEGMENTIFYBUFFER_HPP_
#define SEGMENTIFYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"

#include "components/SegmentBuffer.hpp"

namespace PV {

class SegmentifyBuffer : public ActivityBuffer {
  protected:
   void ioParam_segmentLayerName(enum ParamsIOFlag ioFlag);
   // Defines the way to reduce values within a segment
   // into a single scalar. Options are "average", "sum", and "max".
   void ioParam_inputMethod(enum ParamsIOFlag ioFlag);
   // Defines the way to fill the output segment with the
   // reduced scalar method. Options are "centroid" and "fill"
   void ioParam_outputMethod(enum ParamsIOFlag ioFlag);

  public:
   SegmentifyBuffer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~SegmentifyBuffer();

   char const *getInputMethod() const { return mInputMethod; }
   char const *getOutputMethod() const { return mOutputMethod; }

  protected:
   SegmentifyBuffer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void setOriginalActivity(ObserverTable const *table);
   void setSegmentBuffer(ObserverTable const *table);
   void checkDimensions();

   virtual Response::Status allocateDataStructures() override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;
   void checkLabelValBuf(int newSize);
   void buildLabelToIdx(int batchIdx);
   void calculateLabelVals(int batchIdx);
   void setOutputVals(int batchIdx);

  protected:
   ActivityBuffer *mOriginalActivity = nullptr;
   char *mSegmentLayerName           = nullptr;
   SegmentBuffer *mSegmentBuffer     = nullptr;

   // Reusing this buffer for batches
   // Map to go from label to index into labelVals
   std::map<int, int> mLabelToIdx;
   // Matrix to store values (one dim for features, one for # labels
   int mNumLabelVals  = 0;
   int *mLabelIdxBuf  = nullptr;
   float **mLabelVals = nullptr;
   int **mLabelCount  = nullptr;

   char *mInputMethod  = nullptr;
   char *mOutputMethod = nullptr;

}; // class SegmentifyBuffer

} /* namespace PV */
#endif
