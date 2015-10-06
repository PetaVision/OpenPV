/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "IObjectDetector.hpp"

int main(int argc, char * argv[]) {
   PV_Init * pv_init = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   HyPerCol * hc = build(pv_init);
   int rank = hc->columnId();
   vidint::Image vimage;
   vidint::Roi vroi;
   if (rank==0) {
      vimage.width = 64;
      vimage.height = 64;
      vimage.type = vidint::GRAY8;
      vimage.pPixel = (uint8_t *) calloc((size_t) (vimage.width*vimage.height), sizeof(uint8_t));
      vimage.pPixel[2080] = (uint8_t) 255;
      
      vroi.x=24;
      vroi.y=8;
      vroi.width=32;
      vroi.height=32;
   }
   double simTimeInterval = hc->getStopTime() - hc->getStartTime();
   vidint::IObjectDetector * detector = new vidint::IObjectDetector(hc, "input", simTimeInterval);
   vidint::Object vobject[8];
   uint32_t vmax = 8;
   uint32_t resultCount;
   printf("\n\n\n\nCalling detect with Roi.x=%d, Roi.y=%d:\n", vroi.x, vroi.y);
   detector->detect(&vimage, &vroi, vobject, vmax, &resultCount);
   printf("Detected %u objects!\n", resultCount);

   vroi.x = 16;
   vroi.y = 24;
   printf("\n\n\n\nCalling detect with Roi.x=%d, Roi.y=%d:\n", vroi.x, vroi.y);
   detector->detect(&vimage, &vroi, vobject, vmax, &resultCount);
   printf("Detected %u objects!\n", resultCount);
   delete pv_init;
   return 0;
}
