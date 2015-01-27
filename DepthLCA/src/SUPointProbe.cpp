/*
 * SUPointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "SUPointProbe.hpp"
#include <string.h>

namespace PV {

SUPointProbe::SUPointProbe() {
   initSUPointProbe_base();
   // Default constructor for derived classes.  Derived classes should call initSUPointProbe from their init-method.
}

/**
 * @probeName
 * @hc
 */
SUPointProbe::SUPointProbe(const char * probeName, HyPerCol * hc) :
   PointProbe()
{
   initSUPointProbe_base();
   initPointProbe(probeName, hc);
}

SUPointProbe::~SUPointProbe()
{
}

int SUPointProbe::initSUPointProbe_base() {
   xLoc = 0;
   yLoc = 0;
   fLoc = 0;
   return PV_SUCCESS;
}

int SUPointProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = PointProbe::ioParamsFillGroup(ioFlag);
   ioParam_disparityLayerName(ioFlag);
   ioParam_xLoc(ioFlag);
   ioParam_yLoc(ioFlag);
   ioParam_fLoc(ioFlag);
   return status;
}

void SUPointProbe::ioParam_xLoc(enum ParamsIOFlag ioFlag) {
   //xloc, yloc, and floc are determined from the layer size
}

void SUPointProbe::ioParam_yLoc(enum ParamsIOFlag ioFlag) {

}

void SUPointProbe::ioParam_fLoc(enum ParamsIOFlag ioFlag) {
   
}

void SUPointProbe::ioParam_disparityLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "disparityLayerName", &disparityLayerName);
}

int SUPointProbe::communicateInitInfo() {
   //Target layer set in LayerProbe communicateInitInfo
   int status = LayerProbe::communicateInitInfo();
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   //xLoc is supposed to be extended
   xLoc = ((loc->nxGlobal/2) - 1) + loc->halo.rt;
   yLoc = ((loc->nyGlobal/2) - 1) + loc->halo.up;
   //fLoc changes based on which neuron is being tested, grab from movie layer. Set to 0 for now
   fLoc = 0;

   //Grab disparity movie layer to see which neuron we're testing
   HyPerLayer* h_layer = parent->getLayerFromName(disparityLayerName);
   if (h_layer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: disparityLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, disparityLayerName);
      }
   }
   disparityLayer = dynamic_cast<Movie *>(h_layer);
   if (disparityLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: disparityLayerName \"%s\" is not a MovieLayer.\n",
                 parent->parameters()->groupKeywordFromName(name), name, disparityLayerName);
      }
   }

   return status;
}

int SUPointProbe::outputState(double timef){
   //Initialize with the neuron we're looking at on, with everything else off
   std::string filename = std::string(disparityLayer->getFilename());
   //Parse filename to grab layer name and neuron index
   size_t und_pos = filename.find_last_of("_");
   size_t ext_pos = filename.find_last_of(".");
   fLoc = atoi(filename.substr(und_pos+1, ext_pos-und_pos-1).c_str());
   PointProbe::outputState(timef);
}

/**
 * @time
 * @l
 * @k
 * @kex
 */
int SUPointProbe::writeState(double timef, HyPerLayer * l, int k, int kex) {

   assert(outputstream && outputstream->fp);
   const pvdata_t * V = l->getV();
   const pvdata_t * activity = l->getLayerData();

   fprintf(outputstream->fp, "%s t=%.1f", msg, timef);
   fprintf(outputstream->fp, " feature=%d", fLoc);
   fprintf(outputstream->fp, " a=%.5f", activity[kex]);
   fprintf(outputstream->fp, "\n");
   fflush(outputstream->fp);

   return PV_SUCCESS;
}

} // namespace PV
