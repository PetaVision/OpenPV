#pragma once

#include "InputLayer.hpp"

namespace PV {

   class PvpLayer : public InputLayer {

   protected:
      PvpLayer() {}
      virtual RealBuffer retrieveData(std::string filename, int batchIndex);

   public:
      PvpLayer(const char * name, HyPerCol * hc);
      virtual ~PvpLayer();
      virtual int allocateDataStructures();

   private:
      RealBuffer readSparseBinaryActivityFrame(int numParams, int *params, PV_Stream *pvstream, int frameNumber);
      RealBuffer readSparseValuesActivityFrame(int numParams, int *params, PV_Stream *pvstream, int frameNumber);
      RealBuffer readNonspikingActivityFrame(int numParams, int *params, PV_Stream *pvstream, int frameNumber);
      bool mNeedFrameSizesForSpiking = true;
      std::vector<long> mFrameStartBuffer;
      std::vector<int> mCountBuffer;
      int mPvpFrameCount = -1;
   };
}
