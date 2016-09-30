#ifndef __PVPLAYER_HPP__
#define __PVPLAYER_HPP__

#include "utils/BufferUtilsPvp.hpp"
#include "InputLayer.hpp"

namespace PV {

   class PvpLayer : public InputLayer {

   protected:
      PvpLayer() {}
      virtual Buffer<float> retrieveData(std::string filename, int batchIndex);

   public:
      PvpLayer(const char * name, HyPerCol * hc);
      virtual ~PvpLayer();
      virtual int allocateDataStructures();

   private:
      struct BufferUtils::SparseFileTable sparseTable;
      int mPvpFrameCount = -1;
      int mInputNx  = 0;
      int mInputNy  = 0;
      int mInputNf  = 0;
      int mFileType = 0;
   };
}

#endif
