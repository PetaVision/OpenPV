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
   initialize(probeName, hc);
}

PointProbe::~PointProbe()
{
}

int PointProbe::initPointProbe_base() {
   xLoc = 0;
   yLoc = 0;
   fLoc = 0;
   batchLoc = 0;
   msg = NULL;
   return PV_SUCCESS;
}

int PointProbe::initialize(const char * probeName, HyPerCol * hc) {
   int status = LayerProbe::initialize(probeName, hc);
   return status;
}

int PointProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_xLoc(ioFlag);
   ioParam_yLoc(ioFlag);
   ioParam_fLoc(ioFlag);
   ioParam_batchLoc(ioFlag);
   return status;
}

void PointProbe::ioParam_xLoc(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValueRequired(ioFlag, getName(), "xLoc", &xLoc);
}

void PointProbe::ioParam_yLoc(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValueRequired(ioFlag, getName(), "yLoc", &yLoc);
}

void PointProbe::ioParam_fLoc(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValueRequired(ioFlag, getName(), "fLoc", &fLoc);
}

void PointProbe::ioParam_batchLoc(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValueRequired(ioFlag, getName(), "batchLoc", &batchLoc);
}

int PointProbe::initOutputStream(const char * filename) {
   if(parent->columnId()==0){
      // Called by LayerProbe::initLayerProbe, which is called near the end of PointProbe::initPointProbe
      // So this->xLoc, yLoc, fLoc have been set.
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
      fprintf(stderr, "PointProbe on layer %s: yLoc coordinate %d is out of bounds (layer has %d neurons in the y-direction.\n", getTargetLayer()->getName(), yLoc, loc->nyGlobal);
      status = PV_FAILURE;
   }
   if( (fLoc < 0 || fLoc > loc->nf) && isRoot ) {
      fprintf(stderr, "PointProbe on layer %s: fLoc coordinate %d is out of bounds (layer has %d features.\n", getTargetLayer()->getName(), fLoc, loc->nf);
      status = PV_FAILURE;
   }
   if( (batchLoc < 0 || batchLoc > loc->nbatch) && isRoot ) {
      fprintf(stderr, "PointProbe on layer %s: batchLoc coordinate %d is out of bounds (layer has %d batches.\n", getTargetLayer()->getName(), batchLoc, loc->nbatch);
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
   //We need to calculate which mpi process contains the target point, and send that info to the root process
   //Each process calculates local index
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   //Calculate local cords from global
   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;
   const int kb0 = loc->kb0;
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int nbatch = loc->nbatch;
   const int xLocLocal = xLoc - kx0;
   const int yLocLocal = yLoc - ky0;
   const int nbatchLocal = batchLoc - kb0;
   
   float vval = 0;
   float aval = 0;
   //if in bounds
   if( xLocLocal >= 0 && xLocLocal < nx &&
       yLocLocal >= 0 && yLocLocal < ny &&
       nbatchLocal >= 0 && nbatchLocal < nbatch){
      const pvdata_t * V = getTargetLayer()->getV();
      const pvdata_t * activity = getTargetLayer()->getLayerData();
      //Send V and A to root
      const int k = kIndex(xLocLocal, yLocLocal, fLoc, nx, ny, nf);
      const int kex = kIndexExtended(k, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      if(V){
         vval = V[k + nbatchLocal*getTargetLayer()->getNumNeurons()];
      }
      if(activity){
         aval = activity[kex + nbatchLocal * getTargetLayer()->getNumExtended()];
      }
#ifdef PV_USE_MPI
      //If not in root process, send to root process
      if(parent->columnId()!=0){
         MPI_Send(&vval, 1, MPI_FLOAT, 0, 0, parent->icCommunicator()->communicator());
         MPI_Send(&aval, 1, MPI_FLOAT, 0, 0, parent->icCommunicator()->communicator());
      }
#endif
   }

   //Root process
   if(parent->columnId()==0){
      //Calculate which rank target neuron is
      //TODO we need to calculate rank from batch as well
      int xRank = xLoc/nx;
      int yRank = yLoc/ny;

      int srcRank = rankFromRowAndColumn(yRank, xRank, parent->icCommunicator()->numCommRows(), parent->icCommunicator()->numCommColumns());

#ifdef PV_USE_MPI
      //If srcRank is not root process, MPI_Recv from that rank
      if(srcRank != 0){
         MPI_Recv(&vval, 1, MPI_FLOAT, srcRank, 0, parent->icCommunicator()->communicator(), MPI_STATUS_IGNORE);
         MPI_Recv(&aval, 1, MPI_FLOAT, srcRank, 0, parent->icCommunicator()->communicator(), MPI_STATUS_IGNORE);
      }
#endif
      return point_writeState(timef, vval, aval);
   }
   else{
      return PV_SUCCESS;
   }
}

/**
 * @time
 * @l
 * @k
 * @kex
 */
int PointProbe::point_writeState(double timef, float outVVal, float outAVal) {
   if(parent->columnId()==0){
      assert(outputstream && outputstream->fp);

      fprintf(outputstream->fp, "%s t=%.1f", msg, timef);
      fprintf(outputstream->fp, " V=%6.5f", outVVal);
      fprintf(outputstream->fp, " a=%.5f", outAVal);
      fprintf(outputstream->fp, "\n");
      fflush(outputstream->fp);
   }
   return PV_SUCCESS;
}

} // namespace PV
