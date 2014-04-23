/*
 * MLPOutputLayer.cpp
 * Author: slundquist
 */

#include "MLPOutputLayer.hpp"

namespace PV {
MLPOutputLayer::MLPOutputLayer(){
   initialize_base();
}
MLPOutputLayer::MLPOutputLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

MLPOutputLayer::~MLPOutputLayer(){
   if(classBuffer){
      free(classBuffer);
   }
}

int MLPOutputLayer::initialize_base()
{
   localTarget = true;
   classBuffer = NULL;
   return PV_SUCCESS;
}


int MLPOutputLayer::allocateDataStructures() {
   int status = SigmoidLayer::allocateDataStructures();
   //Allocate buffer size of this layer's nf (number of output classes)
   int nf = getLayerLoc()->nf;
   if(!localTarget){
      classBuffer = (pvdata_t*) malloc(nf * sizeof(pvdata_t));
   }
   return status;
}

int MLPOutputLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = SigmoidLayer::ioParamsFillGroup(ioFlag);
   ioParam_LocalTarget(ioFlag);
   return status;
}

void MLPOutputLayer::ioParam_LocalTarget(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "localTarget", &localTarget, localTarget);
}

int MLPOutputLayer::updateState(double timef, double dt) {
   int status = SigmoidLayer::updateState(timef, dt);
   //If not local, find mean of all output nodes and set to all nodes
   //TODO is mean the right way to do this?

   if(!localTarget){
      const PVLayerLoc * loc = getLayerLoc();
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int numNeurons = getNumNeurons();
      pvdata_t * A = getCLayer()->activity->data;
      assert(classBuffer);
      //Clear classBuffer
      for(int i = 0; i < nf; i++){
         classBuffer[i] = 0;
      }
      //Only go through restricted
      for(int ni = 0; ni < numNeurons; ni++){
         int nExt = kIndexExtended(ni, nx, ny, nf, loc->nb);
         int fi = featureIndex(nExt, nx+2*loc->nb, ny+2*loc->nb, nf);
         //Sum over x and y direction
         classBuffer[fi] += A[nExt];
      }
      //Normalize classBuffer to find mean
      for(int i = 0; i < nf; i++){
         classBuffer[i] /= nx*ny;
      }

      //Reduce all classBuffers through a mean
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, classBuffer, nf, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
      //Normalize classBuffer across processors
      for(int i = 0; i < nf; i++){
         classBuffer[i] /= parent->icCommunicator()->commSize();
      }
#endif // PV_USE_MPI

      //Put new classBuffer into activity
      //Only go through restricted
      for(int ni = 0; ni < numNeurons; ni++){
         int nExt = kIndexExtended(ni, nx, ny, nf, loc->nb);
         int fi = featureIndex(nExt, nx+2*loc->nb, ny+2*loc->nb, nf);
         A[nExt] = classBuffer[fi];
      }
   }

   //For testing purposes
   //for(int ni = 0; ni < getNumNeurons(); ni++){
   //   int nExt = kIndexExtended(ni, nx, ny, nf, loc->nb);
   //   std::cout << timef <<":  ni: " << ni << "  A: " << A[ni] << "\n";
   //}
   return PV_SUCCESS;
}

}
