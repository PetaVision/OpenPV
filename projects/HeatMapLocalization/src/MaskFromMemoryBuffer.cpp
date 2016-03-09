
/*
 * MaskFromMemoryBuffer.cpp
 *
 *  Created on: Feb 18, 2016
 *      Author: peteschultz 
 */

#include "MaskFromMemoryBuffer.hpp"
#include <layers/ImageFromMemoryBuffer.hpp>

MaskFromMemoryBuffer::MaskFromMemoryBuffer(const char * name, PV::HyPerCol * hc){
   initialize_base();
   initialize(name, hc);
}

MaskFromMemoryBuffer::MaskFromMemoryBuffer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

MaskFromMemoryBuffer::~MaskFromMemoryBuffer(){
   free(imageLayerName);
}

int MaskFromMemoryBuffer::initialize_base(){
   imageLayerName = NULL;
   imageLayer = NULL;
   imageLeft = -1;
   imageRight = -1;
   imageTop = -1;
   imageBottom = -1;
   return PV_SUCCESS;
}

int MaskFromMemoryBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_imageLayerName(ioFlag);
   return status;
}

void MaskFromMemoryBuffer::ioParam_imageLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "imageLayerName", &imageLayerName);
}

int MaskFromMemoryBuffer::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();
   HyPerLayer * hyperLayer = parent->getLayerFromName(imageLayerName);
   if (hyperLayer==NULL) {
      if (parent->columnId()==0) {
         fflush(stdout);
         fprintf(stderr, "%s \"%s\" error: imageLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 getKeyword(), name, imageLayerName);
      }
      status = PV_FAILURE;
   }
   else {
      imageLayer = dynamic_cast<PV::ImageFromMemoryBuffer *>(hyperLayer);
      if (imageLayer==NULL) {
         if (parent->columnId()==0) {
         fflush(stdout);
            fprintf(stderr, "%s \"%s\" error: imageLayerName \"%s\" is not an ImageFromMemoryBuffer layer.\n",
                    getKeyword(), name, imageLayerName);
         }
         status = PV_FAILURE;
      }
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (!imageLayer->getInitInfoCommunicatedFlag()) {
      if (parent->columnId()==0) {
         printf("%s \"%s\" must wait until imageLayer \"%s\" has finished its communicateInitInfo stage.\n", getKeyword(), name, imageLayerName);
      }
      return PV_POSTPONE;
   }
   PVLayerLoc const * imageLayerLoc = imageLayer->getLayerLoc();
   PVLayerLoc const * loc = getLayerLoc();
   if (imageLayerLoc->nx != loc->nx || imageLayerLoc->ny != loc->ny) {
      if (parent->columnId()==0) {
         fflush(stdout);
         fprintf(stderr, "%s \"%s\" error: dimensions (%d-by-%d) do not agree with dimensions of image layer \"%s\" (%d-by-%d)n", getKeyword(), name, loc->nx, loc->ny, imageLayerName, imageLayerLoc->nx, imageLayerLoc->ny);
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (status==PV_FAILURE) {
      exit(EXIT_FAILURE);
   }

   return status;
}

int MaskFromMemoryBuffer::updateState(double time, double dt)
{
   if (imageLayer->getImageLeft() == imageLeft &&
       imageLayer->getImageRight() == imageRight &&
       imageLayer->getImageTop() == imageTop &&
       imageLayer->getImageBottom() && imageBottom) {
      return PV_SUCCESS; // mask only needs to change if the imageLayer changes its active region
   }

   imageLeft = imageLayer->getImageLeft();
   imageRight = imageLayer->getImageRight();
   imageTop = imageLayer->getImageTop();
   imageBottom = imageLayer->getImageBottom();

   PVLayerLoc const * loc = getLayerLoc();
   for(int b = 0; b < loc->nbatch; b++){
      pvdata_t * ABatch = getActivity() + b * getNumExtended();
      int const num_neurons = getNumNeurons();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int ni = 0; ni < num_neurons; ni++) {
         PVHalo const * halo = &loc->halo;
         int const nx = loc->nx;
         int const ny = loc->ny;
         int const nf = loc->nf;
         int x = kxPos(ni, nx, ny, nf);
         int y = kyPos(ni, nx, ny, nf);
         pvadata_t a = (pvadata_t) (x>=imageLeft && x < imageRight && y >= imageTop && y < imageBottom);
         int nExt = kIndexExtended(ni, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
         ABatch[nExt] = a;
      }
   }
   return PV_SUCCESS;
}
