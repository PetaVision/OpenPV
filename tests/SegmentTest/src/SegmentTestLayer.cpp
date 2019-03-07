#include "SegmentTestLayer.hpp"
#include <components/ActivityComponent.hpp>

namespace PV {

SegmentTestLayer::SegmentTestLayer(const char *name, PVParams *params, Communicator const *comm) {
   SegmentLayer::initialize(name, params, comm);

   FatalIf(
         mActivityComponent == nullptr,
         "%s failed to create an ActivityComponent.\n",
         getDescription_c());
   mSegmentBuffer = mActivityComponent->getComponentByType<SegmentBuffer>();
   FatalIf(
         mSegmentBuffer == nullptr, "%s failed to create an ActivityBuffer.\n", getDescription_c());
}

Response::Status SegmentTestLayer::checkUpdateState(double timef, double dt) {
   // Do update state first
   SegmentLayer::checkUpdateState(timef, dt);
   const PVLayerLoc *loc = getLayerLoc();

   for (int bi = 0; bi < loc->nbatch; bi++) {
      std::map<int, int> idxMap = mSegmentBuffer->getCenterIdxBuf(bi);
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
