#ifndef __PVPLAYER_HPP__
#define __PVPLAYER_HPP__

#include "InputLayer.hpp"
#include "utils/BufferUtilsPvp.hpp"

namespace PV {

class PvpLayer : public InputLayer {

  protected:
   PvpLayer() {}
   virtual int countInputImages() override;
   virtual Buffer<float> retrieveData(int inputIndex, int batchIndex) override;

  public:
   PvpLayer(const char *name, HyPerCol *hc);
   virtual ~PvpLayer();
   virtual int allocateDataStructures();

  private:
   struct BufferUtils::SparseFileTable sparseTable;
};
}

#endif
