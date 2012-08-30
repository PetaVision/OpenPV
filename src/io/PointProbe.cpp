/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "PointProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include <string.h>

namespace PV {

/**
 * @filename
 * @layer
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointProbe::PointProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc,
      const char * msg) :
   LayerProbe()
{
   initPointProbe(filename, layer, xLoc, yLoc, fLoc, msg);
}

/**
 * @layer
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointProbe::PointProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) :
   LayerProbe()
{
   initPointProbe(NULL, layer, xLoc, yLoc, fLoc, msg);
}

PointProbe::~PointProbe()
{
   free(msg);
}

int PointProbe::initPointProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) {
   int status = PV_SUCCESS;
   const PVLayerLoc * loc = layer->getLayerLoc();
   bool isRoot = layer->getParent()->icCommunicator()->commRank()==0;
   if( (xLoc < 0 || xLoc > loc->nxGlobal) && isRoot ) {
      fprintf(stderr, "PointProbe on layer %s: xLoc coordinate %d is out of bounds (layer has %d neurons in the x-direction.\n", layer->getName(), xLoc, loc->nxGlobal);
      status = PV_FAILURE;
   }
   if( (yLoc < 0 || yLoc > loc->nyGlobal) && isRoot ) {
      fprintf(stderr, "PointProbe on layer %s: yLoc coordinate %d is out of bounds (layer has %d neurons in the y-direction.\n", layer->getName(), xLoc, loc->nyGlobal);
      status = PV_FAILURE;
   }
   if( (fLoc < 0 || fLoc > loc->nf) && isRoot ) {
      fprintf(stderr, "PointProbe on layer %s: fLoc coordinate %d is out of bounds (layer has %d features.\n", layer->getName(), xLoc, loc->nf);
      status = PV_FAILURE;
   }
   if( status != PV_SUCCESS ) abort();
   this->xLoc = xLoc;
   this->yLoc = yLoc;
   this->fLoc = fLoc;
   // this->sparseOutput = false;
   status = initMessage(msg);
   if (status != PV_SUCCESS) abort();
   status = initLayerProbe(filename, layer);
   if (status != PV_SUCCESS) abort();
   return status;
}

int PointProbe::initFilePointer(const char * filename, HyPerLayer * layer) {
   // Called by LayerProbe::initLayerProbe, which is called near the end of PointProbe::initPointProbe
   // So this->xLoc, yLoc, fLoc have been set.
   const PVLayerLoc * loc = layer->getLayerLoc();

   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int xLocLocal = xLoc - kx0;
   const int yLocLocal = yLoc - ky0;
   bool pointInLocalFrame = xLocLocal >= 0 && xLocLocal < nx && yLocLocal >=0 && yLocLocal < ny;
   if (pointInLocalFrame) {
      if( filename != NULL ) {
         char * outputdir = layer->getParent()->getOutputPath();
         char * path = (char *) malloc(strlen(outputdir)+1+strlen(filename)+1);
         sprintf(path, "%s/%s", outputdir, filename);
         fp = fopen(path, "w");
         if( !fp ) {
            fprintf(stderr, "LayerProbe: Unable to open \"%s\" for writing.  Error %d\n", path, errno);
            exit(EXIT_FAILURE);
         }
         free(path);
      }
      else {
         fp = stdout;
      }
   }
   else {
      fp = NULL;
   }
   return PV_SUCCESS;
}

//PointProbe::initMessage and StatsProbe::initMessage are identical.  Move to LayerProbe (even though LayerProbe doesn't use msg?)
int PointProbe::initMessage(const char * msg) {
   int status = PV_SUCCESS;
   if( msg != NULL && msg[0] != '\0' ) {
      size_t msglen = strlen(msg);
      this->msg = (char *) calloc(msglen+2, sizeof(char)); // Allocate room for colon plus null terminator
      if(this->msg) {
         memcpy(this->msg, msg, msglen);
         this->msg[msglen] = ':';
         this->msg[msglen+1] = '\0';
      }
   }
   else {
      this->msg = (char *) calloc(1, sizeof(char));
      if(this->msg) {
         this->msg[0] = '\0';
      }
   }
   if( !this->msg ) {
      fprintf(stderr, "PointProbe: Unable to allocate memory for probe's message.\n");
      status = PV_FAILURE;
   }
   return status;
}

/**
 * @timef
 * NOTES:
 *     - Only the activity buffer covers the extended frame - this is the frame that
 * includes boundaries.
 *     - The other dynamic variables (G_E, G_I, V, Vth) cover the "real" or "restricted"
 *     frame.
 *     - sparseOutput was introduced to deal with ConditionalProbes.
 *     - In MPI runs, xLoc and yLoc refer to global coordinates.
 *     writeState is only called by the processor with (xLoc,yLoc) in its
 *     non-extended region.
 */
int PointProbe::outputState(float timef)
{
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();

   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int xLocLocal = xLoc - kx0;
   const int yLocLocal = yLoc - ky0;
   if( xLocLocal < 0 || xLocLocal >= nx ||
       yLocLocal < 0 || yLocLocal >= ny) return PV_SUCCESS;
   const int nf = loc->nf;

   const int k = kIndex(xLocLocal, yLocLocal, fLoc, nx, ny, nf);
   const int kex = kIndexExtended(k, nx, ny, nf, loc->nb);

   return writeState(timef, getTargetLayer(), k, kex);
}

/**
 * @time
 * @l
 * @k
 * @kex
 */
int PointProbe::writeState(float timef, HyPerLayer * l, int k, int kex) {

   assert(fp);
   const pvdata_t * V = l->getV();
   const pvdata_t * activity = l->getLayerData();

   fprintf(fp, "%s t=%.1f", msg, timef);
   fprintf(fp, " V=%6.5f", V != NULL ? V[k] : 0.0f);
   fprintf(fp, " a=%.5f", activity[kex]);
   fprintf(fp, "\n");
   fflush(fp);

   return PV_SUCCESS;
}

} // namespace PV
