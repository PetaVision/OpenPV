#include "SegmentTestLayer.hpp"

namespace PV {

SegmentTestLayer::SegmentTestLayer(const char *name, HyPerCol *hc) {
   SegmentLayer::initialize(name, hc);
}

Response::Status SegmentTestLayer::updateState(double timef, double dt) {
   // Do update state first
   SegmentLayer::updateState(timef, dt);
   const PVLayerLoc *loc = getLayerLoc();

   for (int bi = 0; bi < loc->nbatch; bi++) {
      std::map<int, int> idxMap = centerIdx[bi];
      for (auto &p : idxMap) {
         int label = p.first;
         int idx   = p.second;
         // Translate idx (global res) into x and y
         int xIdx = idx % loc->nxGlobal;
         int yIdx = idx / loc->nxGlobal;

         int labelX = (label - 1) % 3;
         int labelY = (label - 1) / 3;

         if (labelX == 0) {
            FatalIf(!(xIdx == 1), "Test failed.\n");
         }
         if (labelX == 1) {
            FatalIf(!(xIdx == 4), "Test failed.\n");
         }
         if (labelX == 2) {
            FatalIf(!(xIdx == 6), "Test failed.\n");
         }

         if (labelY == 0) {
            FatalIf(!(yIdx == 1), "Test failed.\n");
         }
         if (labelY == 1) {
            FatalIf(!(yIdx == 4), "Test failed.\n");
         }
         if (labelY == 2) {
            FatalIf(!(yIdx == 6), "Test failed.\n");
         }

         // InfoLog() << "Label " << label << " (" << labelX << ", " << labelY << ") centerpoint: ("
         // << xIdx << ", " << yIdx << ")\n";
      }
   }

   return Response::SUCCESS;
}

} /* namespace PV */
