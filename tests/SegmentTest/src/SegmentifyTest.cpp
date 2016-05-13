#include "SegmentifyTest.hpp"

namespace PV {

SegmentifyTest::SegmentifyTest(const char * name, HyPerCol * hc){
   Segmentify::initialize(name, hc);
}

/*
 * Segment values are such:
 * 1 1 1 2 2 2 3 3
 * 1 1 1 2 2 2 3 3
 * 1 1 1 2 2 2 3 3 
 * 4 4 4 5 5 5 6 6
 * 4 4 4 5 5 5 6 6
 * 4 4 4 5 5 5 6 6
 * 7 7 7 8 8 8 9 9
 * 7 7 7 8 8 8 9 9
 */
float SegmentifyTest::getTargetVal(int yi, int xi, int fi){
   const PVLayerLoc * loc = getLayerLoc();
   //We can convert yi and xi to an index between 0 and 2
   int newYi = yi / 3; 
   int newXi = xi / 3;
   int segmentLabel = newYi * 3 + newXi + 1;
   int returnLabel = -1;
   if(strcmp(inputMethod, "sum") == 0){
      //Account for edge cases
      if(segmentLabel == 3 || segmentLabel == 6 || segmentLabel == 7 || segmentLabel == 8){
         returnLabel = segmentLabel * 6;
      }
      else if(segmentLabel == 9){
         returnLabel = segmentLabel * 4;
      }
      else{
         returnLabel = segmentLabel * 9;
      }
   }
   else if(strcmp(inputMethod, "average") == 0 || strcmp(inputMethod, "max") == 0){
      returnLabel = segmentLabel;
   }
   else{
      //Should never get here
      assert(0);
   }
   return returnLabel;
}

int SegmentifyTest::checkOutputVals(int yi, int xi, int fi, float targetVal, float actualVal){
   const PVLayerLoc * loc = getLayerLoc();
   //We can convert yi and xi to an index between 0 and 2
   int newYi = yi / 3; 
   int newXi = xi / 3;
   int segmentLabel = newYi * 3 + newXi + 1;

   if(strcmp(outputMethod, "centroid") == 0){
      int centX, centY;
      if(newXi == 0){
         centX = 1;
      }
      else if(newXi == 1){
         centX = 4;
      }
      else if(newXi == 2){
         centX = 6;
      }
      if(newYi == 0){
         centY = 1;
      }
      else if(newYi == 1){
         centY = 4;
      }
      else if(newYi == 2){
         centY = 6;
      }

      if(xi == centX && yi == centY){
         assert(actualVal == targetVal);
      }
      else{
         assert(actualVal == 0);
      }
   }
   else if(strcmp(outputMethod, "fill") == 0){
      assert(actualVal == targetVal);
   }
   return PV_SUCCESS;
}

int SegmentifyTest::updateState(double timef, double dt){
   //Do update state first
   Segmentify::updateState(timef, dt);
   const PVLayerLoc * loc = getLayerLoc();
   pvdata_t * A = getActivity();
   assert(A);

   for(int bi = 0; bi < loc->nbatch; bi++){
      pvdata_t * batchA = A + bi * getNumExtended();
      for(int yi = 0; yi < loc->ny; yi++){
         for(int xi = 0; xi < loc->nx; xi++){
            for(int fi = 0; fi < loc->nf; fi++){
               int extIdx = (yi + loc->halo.up) * (loc->nx + loc->halo.lt + loc->halo.rt) * loc->nf + (xi + loc->halo.lt) * loc->nf + fi;
               float actualVal = batchA[extIdx];
               float targetVal = getTargetVal(yi+loc->ky0, xi+loc->kx0, fi);
               checkOutputVals(yi+loc->ky0, xi+loc->kx0, fi, targetVal, actualVal);

               //std::cout << "Idx: (" << bi << "," << yi << "," << xi << "," << fi << ") Val: " << actualVal << " Target: " << targetVal << "\n";
            }
         }
      }

   }

   return PV_SUCCESS;
}

BaseObject * createSegmentifyTest(char const * name, HyPerCol * hc) {
   return hc ? new SegmentifyTest(name, hc) : NULL;
}

} /* namespace PV */
