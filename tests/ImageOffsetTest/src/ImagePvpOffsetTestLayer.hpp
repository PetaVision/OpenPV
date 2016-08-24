#pragma once

#include <layers/PvpLayer.hpp>

namespace PV {
   class ImagePvpOffsetTestLayer: public PV::PvpLayer {
      public:
         ImagePvpOffsetTestLayer(const char* name, HyPerCol * hc);
         virtual double getDeltaUpdateTime();

      protected:
         int updateState(double timef, double dt);
         bool readyForNextFile();
   };
}
