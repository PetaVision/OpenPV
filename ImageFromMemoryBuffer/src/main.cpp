/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "ImageFromMemoryBuffer.hpp"

int custominit(HyPerCol * hc, int argc, char ** argv);
// custominit is for doing initializations after
// HyPerCol, layers, connections, probes etc. have been instantiated
// but before HyPerCol::run is called.

void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().

int main(int argc, char * argv[]) {
   int status;
   status = buildandrun(argc, argv, &custominit, NULL, &customgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int custominit(HyPerCol * hc, int argc, char ** argv) {
   ImageFromMemoryBuffer * ifmb = dynamic_cast<ImageFromMemoryBuffer *>(hc->getLayerFromName("input"));
   assert(ifmb!=NULL);
   int height = ifmb->getLayerLoc()->nyGlobal;
   int width = ifmb->getLayerLoc()->nxGlobal;
   int numbands = ifmb->getLayerLoc()->nf;
   int xstride = numbands;
   int ystride = xstride * width;
   int bandstride = 1;
   
   uint8_t * buffer = NULL;
   if (hc->columnId()==0) {
      buffer = (uint8_t *) calloc((size_t) (height*width*numbands), sizeof(uint8_t));
      if(buffer == NULL) {
         fprintf(stderr, "Unable to allocate a buffer of %d elements of size %zu!\n");
         fprintf(stderr, "Buy another 4K module of RAM or something already!\n");
         exit(EXIT_FAILURE);
      }
      int scintillate = kIndex(height/2,width/2,numbands/2,height,width,numbands);
      buffer[scintillate] = (uint8_t) 255;
   }
   ifmb->setMemoryBuffer(buffer, height, width, numbands, xstride, ystride, bandstride, (uint8_t) 0, (uint8_t) 255);
   assert(buffer==NULL || hc->columnId()==0);
   free(buffer); // ifmb->setMemoryBuffer copies buffer.  This free statement will need to be moved if that stops being the case.
   return PV_SUCCESS;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   if (!strcmp(keyword, "ImageFromMemoryBuffer")) {
      addedGroup = new ImageFromMemoryBuffer(name, hc);
   }
   return addedGroup;
}
