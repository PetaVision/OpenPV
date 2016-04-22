/*
 * WTALayer.cpp
 * Author: slundquist
 */

#include "WTALayer.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

namespace PV {
WTALayer::WTALayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

WTALayer::~WTALayer(){
}

int WTALayer::initialize_base() {
   numChannels = 0;
   originalLayerName = NULL;
   originalLayer = NULL;
   binMax = 1;
   binMin = 0;
   return PV_SUCCESS;
}

int WTALayer::communicateInitInfo() {
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
   if (originalLayer->getInitInfoCommunicatedFlag()==false) {
      return PV_POSTPONE;
   }
   originalLayer->synchronizeMarginWidth(this);
   this->synchronizeMarginWidth(originalLayer);
   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * loc = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" does not have the same dimensions.\n",
                 getKeyword(), name, originalLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d) versus (nx=%d, ny=%d)\n",
                 srcLoc->nxGlobal, srcLoc->nyGlobal, loc->nxGlobal, loc->nyGlobal);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if(getLayerLoc()->nf != 1){
      fprintf(stderr, "%s \"%s\" error: WTALayer can only have 1 feature.\n",
         getKeyword(), name);
   }
   assert(srcLoc->nx==loc->nx && srcLoc->ny==loc->ny);
   return status;
}

int WTALayer::allocateV(){
   //Allocate V does nothing since binning does not need a V layer
   clayer->V = NULL;
   return PV_SUCCESS;
}

int WTALayer::initializeV() {
   assert(getV() == NULL);
   return PV_SUCCESS;
}

int WTALayer::initializeActivity() {
   return PV_SUCCESS;
}

int WTALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_originalLayerName(ioFlag);
   ioParam_binMaxMin(ioFlag);
   return status;
}
void WTALayer::ioParam_originalLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalLayerName", &originalLayerName);
   assert(originalLayerName);
   if (ioFlag==PARAMS_IO_READ && originalLayerName[0]=='\0') {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName must be set.\n",
                 getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void WTALayer::ioParam_binMaxMin(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "binMax", &binMax, binMax);
   parent->ioParamValue(ioFlag, name, "binMin", &binMin, binMin);
   if(ioFlag == PARAMS_IO_READ && binMax <= binMin){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: binMax (%f) must be greater than binMin (%f).\n",
            getKeyword(), name, binMax, binMin);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

int WTALayer::updateState(double timef, double dt) {
   pvdata_t * currA = getCLayer()->activity->data;
   pvdata_t * srcA = originalLayer->getCLayer()->activity->data;

   const PVLayerLoc * loc = getLayerLoc(); 
   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();

   //NF must be one
   assert(loc->nf == 1);
   float stepSize = (float)(binMax - binMin)/(float)srcLoc->nf;

   for(int b = 0; b < srcLoc->nbatch; b++){
      pvdata_t * currABatch = currA + b * getNumExtended();
      pvdata_t * srcABatch = srcA + b * originalLayer->getNumExtended();
      //Loop over x and y of the src layer
      for(int yi = 0; yi < srcLoc->ny; yi++){
         for(int xi = 0; xi < srcLoc->nx; xi++){
            //Find maximum output value in nf direction
            float maxInput = -99999999;
            float maxIndex = -99999999;
            for(int fi = 0; fi < srcLoc->nf; fi++){
               int kExt = kIndex(xi, yi, fi,
                     srcLoc->nx+srcLoc->halo.lt+srcLoc->halo.rt,
                     srcLoc->ny+srcLoc->halo.up+srcLoc->halo.dn,
                     srcLoc->nf);
               if(srcABatch[kExt] > maxInput || fi == 0){
                  maxInput = srcABatch[kExt];
                  maxIndex = fi;
               }
            }
            //Found max index, set output to value
            float outVal = (maxIndex * stepSize) + binMin;
            int kExtOut = kIndex(xi, yi, 0,
                     loc->nx+loc->halo.lt+loc->halo.rt,
                     loc->ny+loc->halo.up+loc->halo.dn,
                     loc->nf);
            currABatch[kExtOut] = outVal;
         }
      }
   }
   return PV_SUCCESS;
}

BaseObject * createWTALayer(char const * name, HyPerCol * hc) {
   return hc ? new WTALayer(name, hc) : NULL;
}

}
