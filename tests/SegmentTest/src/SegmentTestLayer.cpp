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
      for(std::map<int, int>::iterator it = idxMap.begin();
            it != idxMap.end(); ++it){
         int label = it->first;
         int idx = it->second;
         //Translate idx (global res) into x and y
         int xIdx = idx % loc->nxGlobal;
         int yIdx = idx / loc->nxGlobal;

         int labelX = (label-1)% 3;
         int labelY = (label-1)/ 3;

         if(labelX == 0){
            assert(xIdx == 1);
         }
         if(labelX == 1){
            assert(xIdx == 4);
         }
         if(labelX == 2){
            assert(xIdx == 6);
         }

         if(labelY == 0){
            assert(yIdx == 1);
         }
         if(labelY == 1){
            assert(yIdx == 4);
         }
         if(labelY == 2){
            assert(yIdx == 6);
         }

         //std::cout << "Label " << label << " (" << labelX << ", " << labelY << ") centerpoint: (" << xIdx << ", " << yIdx << ")\n";
      }
   }



   


   return PV_SUCCESS;
}

BaseObject * createSegmentTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new SegmentTestLayer(name, hc) : NULL;
}

} /* namespace PV */
