/*
 * BackgroundLayer.cpp
 *
 *  Created on: 4/16/15
 *  slundquist
 */

#include "BackgroundLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

namespace PV {
BackgroundLayer::BackgroundLayer() {
   initialize_base();
}

BackgroundLayer::BackgroundLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

BackgroundLayer::~BackgroundLayer()
{
}

int BackgroundLayer::initialize_base() {
   originalLayer = NULL;
   repFeatureNum = 1;
   return PV_SUCCESS;
}

int BackgroundLayer::initialize(const char * name, HyPerCol * hc) {
   //int num_channels = sourceLayer->getNumChannels();
   int status_init = CloneVLayer::initialize(name, hc);

   return status_init;
}

int BackgroundLayer::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();
   originalLayer = parent->getLayerFromName(originalLayerName);
   if (originalLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 getKeyword(), name, originalLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   //originalLayer->synchronizeMarginWidth(this);
   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * loc = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" does not have the same X/Y dimensions.\n",
                 getKeyword(), name, originalLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                 srcLoc->nxGlobal, srcLoc->nyGlobal, srcLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if ((srcLoc->nf + 1)*repFeatureNum != loc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: nf must have (n+1)*repFeatureNum (%d) features in BackgroundLayer \"%s\", where n is the orig layer number of features.\n",
                 getKeyword(), name, (srcLoc->nf+1)*repFeatureNum, originalLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                 srcLoc->nxGlobal, srcLoc->nyGlobal, srcLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   assert(srcLoc->nx==loc->nx && srcLoc->ny==loc->ny);
   return status;
}

//Background Layer does not use the V buffer, so absolutely fine to clone off of an null V layer
int BackgroundLayer::allocateV() {
   //Do nothing
   return PV_SUCCESS;
}

void BackgroundLayer::ioParam_repFeatureNum(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "repFeatureNum", &repFeatureNum, repFeatureNum);
   if(repFeatureNum <= 0){
      std::cout << "BackgroundLayer " << name << " error: repFeatureNum must an integer greater or equal to 1 (1 feature means no replication)\n";
      exit(-1);
   }
}

int BackgroundLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag){
  //readOriginalLayerName(params);  // done in CloneVLayer
   CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_repFeatureNum(ioFlag);
   return PV_SUCCESS;
}

int BackgroundLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtendedAllBatches);
   return 0;
}

int BackgroundLayer::updateState(double timef, double dt) {
   int status = PV_SUCCESS;
   //int numNeurons = originalLayer->getNumNeurons();
   pvdata_t * A = clayer->activity->data;
   const pvdata_t * originalA = originalLayer->getCLayer()->activity->data;
   const PVLayerLoc * loc = getLayerLoc();
   const PVLayerLoc * locOriginal = originalLayer->getLayerLoc();

   //Make sure all sizes match
   assert(locOriginal->nx == loc->nx);
   assert(locOriginal->ny == loc->ny);
   assert((locOriginal->nf + 1)*repFeatureNum == loc->nf);

   int nx = loc->nx;
   int ny = loc->ny;
   int origNf = locOriginal->nf;
   int thisNf = loc->nf;
   int nbatch = loc->nbatch;

   PVHalo const * halo = &loc->halo;
   PVHalo const * haloOrig = &locOriginal->halo;

   for(int b = 0; b < nbatch; b++){
      pvdata_t * ABatch = A + b * getNumExtended();
      const pvdata_t * originalABatch = originalA + b * originalLayer->getNumExtended();

      //Loop through all nx and ny
      // each y value specifies a different target so ok to thread here (sum, sumsq are defined inside loop)
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int iY = 0; iY < ny; iY++){ 
         for(int iX = 0; iX < nx; iX++){ 
            //outVal stores the NOR of the other values
            int outVal = 1;
            //Shift all features down by one
            for(int iF = 0; iF < origNf; iF++){
               int kextOrig = kIndex(iX, iY, iF, nx+haloOrig->lt+haloOrig->rt, ny+haloOrig->dn+haloOrig->up, origNf);
               float origActivity = originalABatch[kextOrig];
               //outVal is the final out value for the background
               if(origActivity != 0){
                  outVal = 0;
               }
               //Loop over replicated features
               for(int repIdx = 0; repIdx < repFeatureNum; repIdx++){
                  //Index iF one down, multiply by replicate feature number, add repIdx offset
                  int newFeatureIdx = ((iF+1)*repFeatureNum) + repIdx;
                  assert(newFeatureIdx < thisNf);
                  int kext = kIndex(iX, iY, newFeatureIdx, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, thisNf);
                  ABatch[kext] = origActivity;
               }
            }
            //Set background indices to outVal
            for(int repIdx = 0; repIdx < repFeatureNum; repIdx++){
               int kextBackground = kIndex(iX, iY, repIdx, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, thisNf);
               ABatch[kextBackground] = outVal;
            }
         }
      }
   }
   return status;
}

BaseObject * createBackgroundLayer(char const * name, HyPerCol * hc) {
   return hc ? new BackgroundLayer(name, hc) : NULL;
}

} // end namespace PV

