#include "SegmentTestLayer.hpp"

namespace PV {

SegmentTestLayer::SegmentTestLayer(const char * name, HyPerCol * hc){
   SegmentLayer::initialize(name, hc);
}

int SegmentTestLayer::updateState(double timef, double dt){
   //Do update state first
   SegmentLayer::updateState(timef, dt);
   const PVLayerLoc * loc = getLayerLoc();

   for(int bi = 0; bi < loc->nbatch; bi++){
      std::map<int, int> idxMap = centerIdx[bi];
      for(auto& p : idxMap) {
         int label = p.first;
         int idx = p.second;
         //Translate idx (global res) into x and y
         int xIdx = idx % loc->nxGlobal;
         int yIdx = idx / loc->nxGlobal;

         int labelX = (label-1)% 3;
         int labelY = (label-1)/ 3;

         if(labelX == 0){
            pvErrorIf(!(xIdx == 1), "Test failed.\n");
         }
         if(labelX == 1){
            pvErrorIf(!(xIdx == 4), "Test failed.\n");
         }
         if(labelX == 2){
            pvErrorIf(!(xIdx == 6), "Test failed.\n");
         }

         if(labelY == 0){
            pvErrorIf(!(yIdx == 1), "Test failed.\n");
         }
         if(labelY == 1){
            pvErrorIf(!(yIdx == 4), "Test failed.\n");
         }
         if(labelY == 2){
            pvErrorIf(!(yIdx == 6), "Test failed.\n");
         }

         //pvInfo() << "Label " << label << " (" << labelX << ", " << labelY << ") centerpoint: (" << xIdx << ", " << yIdx << ")\n";
      }
   }



   


   return PV_SUCCESS;
}

} /* namespace PV */
