
/*
 * MaskFromMemoryBuffer.cpp
 *
 *  Created on: Feb 18, 2016
 *      Author: peteschultz 
 */

#include "MaskFromMemoryBuffer.hpp"
#include <layers/BaseInput.hpp>

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
   dataLeft = -1;
   dataRight = -1;
   dataTop = -1;
   dataBottom = -1;
   return PV_SUCCESS;
}

int MaskFromMemoryBuffer::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_imageLayerName(ioFlag);
   return status;
}

void MaskFromMemoryBuffer::ioParam_imageLayerName(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "imageLayerName", &imageLayerName);
}

int MaskFromMemoryBuffer::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();
   HyPerLayer * hyperLayer = parent->getLayerFromName(imageLayerName);
   if (hyperLayer==NULL) {
      if (parent->columnId()==0) {
         ErrorLog().printf("%s: imageLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 getDescription_c(), imageLayerName);
      }
      status = PV_FAILURE;
   }
   else {
      imageLayer = dynamic_cast<PV::BaseInput *>(hyperLayer);
      if (imageLayer==NULL) {
         if (parent->columnId()==0) {
         ErrorLog().printf("%s: imageLayerName \"%s\" is not an BaseInput-derived layer.\n",
               getDescription_c(), imageLayerName);
         }
         status = PV_FAILURE;
      }
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   if (!imageLayer->getInitInfoCommunicatedFlag()) {
      if (parent->columnId()==0) {
         InfoLog().printf("%s must wait until imageLayer \"%s\" has finished its communicateInitInfo stage.\n", getDescription_c(), imageLayerName);
      }
      return PV_POSTPONE;
   }
   PVLayerLoc const * imageLayerLoc = imageLayer->getLayerLoc();
   PVLayerLoc const * loc = getLayerLoc();
   if (imageLayerLoc->nx != loc->nx || imageLayerLoc->ny != loc->ny) {
      if (parent->columnId()==0) {
         ErrorLog().printf("%s: dimensions (%d-by-%d) do not agree with dimensions of image layer \"%s\" (%d-by-%d)n", getDescription_c(), loc->nx, loc->ny, imageLayerName, imageLayerLoc->nx, imageLayerLoc->ny);
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   if (status==PV_FAILURE) {
      exit(EXIT_FAILURE);
   }

   return status;
}

int MaskFromMemoryBuffer::updateState(double time, double dt)
{
   if (imageLayer->getDataLeft() == dataLeft &&
         imageLayer->getDataTop() == dataTop &&
         imageLayer->getDataWidth() == dataRight-dataLeft &&
         imageLayer->getDataHeight() && dataBottom-dataTop) {
      return PV_SUCCESS; // mask only needs to change if the imageLayer changes its active region
   }

   dataLeft = imageLayer->getDataLeft();
   dataRight = dataLeft+imageLayer->getDataWidth();
   dataTop = imageLayer->getDataTop();
   dataBottom = dataTop + imageLayer->getDataHeight();

   PVLayerLoc const * loc = getLayerLoc();
   for(int b = 0; b < loc->nbatch; b++){
      float * ABatch = getActivity() + b * getNumExtended();
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
         float a = (float) (x>=dataLeft && x < dataRight && y >= dataTop && y < dataBottom);
         int nExt = kIndexExtended(ni, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
         ABatch[nExt] = a;
      }
   }
   return PV_SUCCESS;
}

PV::BaseObject * createMaskFromMemoryBuffer(char const * name, PV::HyPerCol * hc) {
   return hc ? new MaskFromMemoryBuffer(name, hc) : NULL;
}
