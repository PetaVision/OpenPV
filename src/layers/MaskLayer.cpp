
/*
 * MaskLayer.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist 
 */

#include "MaskLayer.hpp"

namespace PV {

MaskLayer::MaskLayer(const char * name, HyPerCol * hc){
   initialize_base();
   initialize(name, hc);
}

MaskLayer::MaskLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

MaskLayer::~MaskLayer(){
   if(maskLayerName){
      free(maskLayerName);
   }
}

int MaskLayer::initialize_base(){
   maskLayerName = NULL;
   maskLayer = NULL;
   return PV_SUCCESS;
}

int MaskLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_maskLayerName(ioFlag);
   return status;
}

void MaskLayer::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "maskLayerName", &maskLayerName);
}

int MaskLayer::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();
   maskLayer = parent->getLayerFromName(maskLayerName);
   if (maskLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, maskLayerName);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }

   const PVLayerLoc * maskLoc = maskLayer->getLayerLoc();
   const PVLayerLoc * loc = getLayerLoc();
   assert(maskLoc != NULL && loc != NULL);
   if (maskLoc->nxGlobal != loc->nxGlobal || maskLoc->nyGlobal != loc->nyGlobal) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" does not have the same x and y dimensions.\n",
                 parent->parameters()->groupKeywordFromName(name), name, maskLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                 maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }

   if(maskLoc->nf != 1 && maskLoc->nf != loc->nf){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" must either have the same number of features as this layer, or one feature.\n",
                 parent->parameters()->groupKeywordFromName(name), name, maskLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                 maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }

   assert(maskLoc->nx==loc->nx && maskLoc->ny==loc->ny);

   return status;
}

int MaskLayer::updateState(double time, double dt)
{
   ANNLayer::updateState(time, dt);
   const PVLayerLoc * loc = getLayerLoc();
   const PVLayerLoc * maskLoc = maskLayer->getLayerLoc();
   pvdata_t * maskActivity = maskLayer->getActivity();
   pvdata_t * A = getActivity();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for(int ni = 0; ni < num_neurons; ni++){
      int kThisRes = ni;
      int kThisExt = kIndexExtended(ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      int kMaskRes;
      if(maskLoc->nf == 1){
         kMaskRes = ni/nf;
      }
      else{
         kMaskRes = ni;
      }
      int kMaskExt = kIndexExtended(ni, nx, ny, maskLoc->nf, maskLoc->halo.lt, maskLoc->halo.rt, maskLoc->halo.dn, maskLoc->halo.up);

      //Set value to 0, otherwise, updateState from ANNLayer should have taken care of it
      if(maskActivity[kMaskExt] == 0){
         A[kThisExt] = 0;
      }
   }
   return PV_SUCCESS;
}

} /* namespace PV */
