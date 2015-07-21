
/*
 * HeliTileGTLayer.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist 
 */

#include "HeliTileGTLayer.hpp"

namespace PV {

HeliTileGTLayer::HeliTileGTLayer()
{
   initialize_base();
}

HeliTileGTLayer::HeliTileGTLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

HeliTileGTLayer::~HeliTileGTLayer()
{
   if(inputTileLayer) free(inputTileLayer);
   inputLayer = NULL;
}

int HeliTileGTLayer::initialize_base()
{
   inputTileLayer = NULL;
   return PV_SUCCESS;
}

int HeliTileGTLayer::communicateInitInfo(){
   int status = ANNLayer::communicateInitInfo();
   HyPerLayer* tmpLayer = parent->getLayerFromName(inputTileLayer);
   if (tmpLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: inputTileLayer \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, inputTileLayer);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   inputLayer = dynamic_cast<HeliTileMovie*>(tmpLayer);
   if (inputLayer ==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: inputTileLayer \"%s\" is not a HeliTileMovie.\n",
                 parent->parameters()->groupKeywordFromName(name), name, inputTileLayer);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   return status;
}

int HeliTileGTLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_InputTileLayer(ioFlag);
   return status;
}

void HeliTileGTLayer::ioParam_InputTileLayer(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "inputTileLayer", &inputTileLayer);
}

int HeliTileGTLayer::updateState(double time, double dt)
{
   pvdata_t * A = getCLayer()->activity->data;
   const PVLayerLoc * loc = getLayerLoc(); 
   //Only one feature allowed
   assert(loc->nf == 1);
   for(int ni = 0; ni < getNumNeurons(); ni++){
      int nExt = kIndexExtended(ni, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      A[nExt] = inputLayer->getGroundTruth();
      std::cout << "Time: " << time << " Ground Truth: " << inputLayer->getGroundTruth() << "\n";
   }
   return PV_SUCCESS;
}

} /* namespace PV */
