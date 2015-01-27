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

PointProbe::PointProbe() {
   initPointProbe_base();
   // Default constructor for derived classes.  Derived classes should call initPointProbe from their init-method.
}

/**
 * @probeName
 * @hc
 */
PointProbe::PointProbe(const char * probeName, HyPerCol * hc) :
   LayerProbe()
{
   initPointProbe_base();
   initPointProbe(probeName, hc);
}

PointProbe::~PointProbe()
{
}

int PointProbe::initPointProbe_base() {
   xLoc = 0;
   yLoc = 0;
   fLoc = 0;
   msg = NULL;
   return PV_SUCCESS;
}

int PointProbe::initPointProbe(const char * probeName, HyPerCol * hc) {
   int status = LayerProbe::initialize(probeName, hc);
   return status;
}

int PointProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_xLoc(ioFlag);
   ioParam_yLoc(ioFlag);
   ioParam_fLoc(ioFlag);
   return status;
}

void PointProbe::ioParam_xLoc(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValueRequired(ioFlag, getName(), "xLoc", &xLoc);
}

void PointProbe::ioParam_yLoc(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValueRequired(ioFlag, getName(), "yLoc", &xLoc);
}

void PointProbe::ioParam_fLoc(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValueRequired(ioFlag, getName(), "fLoc", &xLoc);
}

int PointProbe::initOutputStream(const char * filename) {
   // Called by LayerProbe::initLayerProbe, which is called near the end of PointProbe::initPointProbe
   // So this->xLoc, yLoc, fLoc have been set.
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   if( filename != NULL ) {
      char * outputdir = getParent()->getOutputPath();
      char * path = (char *) malloc(strlen(outputdir)+1+strlen(filename)+1);
      sprintf(path, "%s/%s", outputdir, filename);
      outputstream = PV_fopen(path, "w", false/*verifyWrites*/);
      if( !outputstream ) {
         fprintf(stderr, "LayerProbe error opening \"%s\" for writing: %s\n", path, strerror(errno));
         exit(EXIT_FAILURE);
      }
      free(path);
   }
   else {
      outputstream = PV_stdout();
   }
   return PV_SUCCESS;
}

int PointProbe::communicateInitInfo() {
   int status = LayerProbe::communicateInitInfo();
   assert(getTargetLayer());
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   bool isRoot = getParent()->icCommunicator()->commRank()==0;
   if( (xLoc < 0 || xLoc > loc->nxGlobal) && isRoot ) {
      fprintf(stderr, "PointProbe on layer %s: xLoc coordinate %d is out of bounds (layer has %d neurons in the x-direction.\n", getTargetLayer()->getName(), xLoc, loc->nxGlobal);
      status = PV_FAILURE;
   }
   if( (yLoc < 0 || yLoc > loc->nyGlobal) && isRoot ) {
      fprintf(stderr, "PointProbe on layer %s: yLoc coordinate %d is out of bounds (layer has %d neurons in the y-direction.\n", getTargetLayer()->getName(), xLoc, loc->nyGlobal);
      status = PV_FAILURE;
   }
   if( (fLoc < 0 || fLoc > loc->nf) && isRoot ) {
      fprintf(stderr, "PointProbe on layer %s: fLoc coordinate %d is out of bounds (layer has %d features.\n", getTargetLayer()->getName(), xLoc, loc->nf);
      status = PV_FAILURE;
   }
   if( status != PV_SUCCESS ) abort();
   return status;
}

/**
 * @timef
 * NOTES:
 *     - Only the activity buffer covers the extended frame - this is the frame that
 * includes boundaries.
 *     - The membrane potential V covers the "real" or "restricted" frame.
 *     - In MPI runs, xLoc and yLoc refer to global coordinates.
 *     writeState is only called by the processor with (xLoc,yLoc) in its
 *     non-extended region.
 */
int PointProbe::outputState(double timef)
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
   const int kex = kIndexExtended(k, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);

   return writeState(timef, getTargetLayer(), k, kex);
}

/**
 * @time
 * @l
 * @k
 * @kex
 */
int PointProbe::writeState(double timef, HyPerLayer * l, int k, int kex) {

   assert(outputstream && outputstream->fp);
   const pvdata_t * V = l->getV();
   const pvdata_t * activity = l->getLayerData();

   fprintf(outputstream->fp, "%s t=%.1f", msg, timef);
   fprintf(outputstream->fp, " V=%6.5f", V != NULL ? V[k] : 0.0f);
   fprintf(outputstream->fp, " a=%.5f", activity[kex]);
   fprintf(outputstream->fp, "\n");
   fflush(outputstream->fp);

   return PV_SUCCESS;
}

} // namespace PV
