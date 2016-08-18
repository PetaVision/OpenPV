#ifndef PVPLAYER_HPP_
#define PVPLAYER_HPP_

#include "InputLayer.hpp"


namespace PV {

   class PvpLayer : public InputLayer {

   protected:
      PvpLayer();
      int initialize(const char * name, HyPerCol * hc);
      virtual Buffer retrieveData(std::string filename, int batchIndex);
      virtual int postProcess(double timef, double dt);
      virtual bool readyForNextFile();

   public:
      PvpLayer(const char * name, HyPerCol * hc);
      virtual ~PvpLayer();

   private:
      Buffer readSparseBinaryActivityFrame(int numParams, int *params, PV_Stream *pvstream, int frameNumber);
      Buffer readSparseValuesActivityFrame(int numParams, int *params, PV_Stream *pvstream, int frameNumber);
      Buffer readNonspikingActivityFrame(int numParams, int *params, PV_Stream *pvstream, int frameNumber);
      int initialize_base();
      bool mInitializedBatchIndexer = false;
      bool mNeedFrameSizesForSpiking = true;
      std::vector<long> mFrameStartBuffer; // What are these used for?
      std::vector<int> mCountBuffer;
   };
}

#endif 
