#pragma once

#include "InputLayer.hpp"

namespace PV {

   class PvpLayer : public InputLayer {

   protected:
      PvpLayer() {}
      virtual Buffer retrieveData(std::string filename, int batchIndex);

   public:
      PvpLayer(const char * name, HyPerCol * hc);
      virtual ~PvpLayer();
      virtual int allocateDataStructures();

   private:
      Buffer readSparseBinaryActivityFrame(int numParams, int *params, PV_Stream *pvstream, int frameNumber);
      Buffer readSparseValuesActivityFrame(int numParams, int *params, PV_Stream *pvstream, int frameNumber);
      Buffer readNonspikingActivityFrame(int numParams, int *params, PV_Stream *pvstream, int frameNumber);
      bool mNeedFrameSizesForSpiking = true;
      std::vector<long> mFrameStartBuffer;
      std::vector<int> mCountBuffer;
      int mPvpFrameCount = -1;
   };
}
