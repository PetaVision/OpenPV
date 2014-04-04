/*
 * HyPerLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCALayer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void HyPerLCALayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int nb,
    const int numChannels,

    float * V,
    const float Vth,
    const float AMax,
    const float AMin,
    const float AShift,
    const float VWidth,
    const float tau_max,
    const float tau_min,
    const float slope_error_std,
    bool selfInteract,
    float * dt_tau,
    float * GSynHead,
    float * activity,
    double * error_mean,
    double * error_std);

#ifdef __cplusplus
}
#endif

namespace PV {

HyPerLCALayer::HyPerLCALayer()
{
   initialize_base();
}

HyPerLCALayer::HyPerLCALayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

HyPerLCALayer::~HyPerLCALayer()
{
}

int HyPerLCALayer::initialize_base()
{
   numChannels = 1; // If a connection connects to this layer on inhibitory channel, HyPerLayer::requireChannel will add necessary channel
   tauMax = 1.0;
   tauMin = tauMax;
   errorStd = 1.0;
   slopeErrorStd = 1.0;
   //Locality in KernelConn
   numWindowX = 1;
   numWindowY = 1;
   windowSymX = false;
   windowSymY = false;
   selfInteract = true;
   dVThresh = 0;
   sparseProbe = NULL;
   targetSparsity = 0;
   return PV_SUCCESS;
}

bool HyPerLCALayer::inWindowExt(int windowId, int neuronIdxExt){
   const PVLayerLoc * loc = this->getLayerLoc();
   int globalExtX = kxPos(neuronIdxExt, loc->nx + 2*loc->nb, loc->ny + 2*loc->nb, loc->nf) + loc->kx0;
   int globalExtY = kyPos(neuronIdxExt, loc->nx + 2*loc->nb, loc->ny + 2*loc->nb, loc->nf) + loc->ky0;
   int outWindow = calcWindow(globalExtX, globalExtY);
   //std::cout << globalExtX << "/" << loc->nxGlobal + 2*loc->nb << " " << globalExtY << "/" << loc->nyGlobal + 2*loc->nb << " " << outWindow << "," << windowId << "\n";
   return (outWindow == windowId);
}

bool HyPerLCALayer::inWindowRes(int windowId, int neuronIdxRes){
   const PVLayerLoc * loc = this->getLayerLoc();
   int globalExtX = kxPos(neuronIdxRes, loc->nx, loc->ny, loc->nf) + loc->kx0;
   int globalExtY = kyPos(neuronIdxRes, loc->nx, loc->ny, loc->nf) + loc->ky0;
   int outWindow = calcWindow(globalExtX, globalExtY);
   return (outWindow == windowId);
}

int HyPerLCALayer::calcWindow(int globalExtX, int globalExtY){
   const PVLayerLoc * loc = this->getLayerLoc();
   //Calculate x and y with symmetry on
   if(windowSymX && globalExtX >= floor((loc->nxGlobal + 2*loc->nb)/2)){
      globalExtX = loc->nxGlobal+2*loc->nb - globalExtX - 1;
   }
   if(windowSymY && globalExtY >= floor((loc->nyGlobal + 2*loc->nb)/2)){
      globalExtY = loc->nyGlobal+2*loc->nb - globalExtY - 1;
   }
   //Calculate the window x and y
   int windowX = floor(((float)globalExtX/(loc->nxGlobal+2*loc->nb)) * numWindowX); 
   int windowY = floor(((float)globalExtY/(loc->nyGlobal+2*loc->nb)) * numWindowY);
   //Change x and y into index
   int windowIdx = (windowY * numWindowY) + windowX;
   assert(windowIdx < getNumWindows());
   assert(windowIdx >= 0);
   return windowIdx;
}

int HyPerLCALayer::getNumWindows(){
   int windowsX, windowsY;
   if(windowSymX){
      windowsX = ceil((float)numWindowX / 2);
   }
   else{
      windowsX = numWindowX;
   }
   if(windowSymY){
      windowsY = ceil((float)numWindowY / 2);
   }
   else{
      windowsY = numWindowY;
   }
   return windowsX * windowsY;
}

int HyPerLCALayer::initialize(const char * name, HyPerCol * hc)
{
   ANNLayer::initialize(name, hc);
   return PV_SUCCESS;
}

int HyPerLCALayer::allocateDataStructures(){
   int status = ANNLayer::allocateDataStructures();
   if(dVThresh != 0){
      //Look for the sparsityLayerProbe
      //Need to do this in allocate since the probe will get attached in communicate
      for(int i = 0; i < numProbes; i++){
         sparseProbe = dynamic_cast<SparsityLayerProbe*>(probes[i]);
         if(sparseProbe){
            break;
         }
      }
      if(!sparseProbe){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" A SparsityLayerProbe must be attached to the layer for a dynamic VThresh (dVThresh != 0)\n",
                  parent->parameters()->groupKeywordFromName(name), name);
         }
#if PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
      targetSparsity= sparseProbe->getInitSparsityVal();
      if(targetSparsity == 0){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" Target Sparsity (initSparsityVal in SparsityLayerProbe) cannot equal 0\n",
                  parent->parameters()->groupKeywordFromName(name), name);
         }
#if PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
   }
   return status;
}

int HyPerLCALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_numChannels(ioFlag);
   ioParam_timeConstantTau(ioFlag);
   ioParam_timeConstantTauMinimum(ioFlag);
   ioParam_numWindowX(ioFlag);
   ioParam_numWindowY(ioFlag);
   ioParam_windowSymX(ioFlag);
   ioParam_windowSymY(ioFlag);
   ioParam_slopeErrorStd(ioFlag);
   ioParam_selfInteract(ioFlag);
   ioParam_dVThresh(ioFlag);
   return status;
}

void HyPerLCALayer::ioParam_dVThresh(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dVThresh", &dVThresh, dVThresh, false/*warnIfAbsent*/);
}

//void HyPerLCALayer::ioParam_targetSparsity(enum ParamsIOFlag ioFlag){
//   assert(!parent->parameters()->presentAndNotBeenRead(name, "dVThresh"));
//   if(dVThresh != 0){
//      if(ioFlag == IO_PARAMS_READ){
//         if(parent->parameters()->present(name, "targetSparsity")){
//            if(!parent->parameters()->present(name, "dVThresh")){
//               if (parent->columnId()==0) {
//                  fprintf(stderr, "%s \"%s\": dVThresh must be specified for targetSparsity\n",
//                        parent->parameters()->groupKeywordFromName(name), name);
//               }
//#if PV_USE_MPI
//               MPI_Barrier(parent->icCommunicator()->communicator());
//#endif
//               exit(EXIT_FAILURE);
//            }
//         }
//      }
//      parent->ioParamValue(ioFlag, name, "targetSparsity", &targetSparsity, targetSparsity, false/*warnIfAbsent*/);
//   }
//}

void HyPerLCALayer::ioParam_numChannels(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "numChannels", &numChannels, numChannels, true/*warnIfAbsent*/);
   if (numChannels != 1 && numChannels != 2){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" requires 1 or 2 channels, numChannels = %d\n",
               parent->parameters()->groupKeywordFromName(name), name, numChannels);
      }
#if PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
}

void HyPerLCALayer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "timeConstantTau", &tauMax, tauMax, true/*warnIfAbsent*/);
}

void HyPerLCALayer::ioParam_timeConstantTauMinimum(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "timeConstantTau"));
   parent->ioParamValue(ioFlag, name, "timeConstantTauMinimum", &tauMin, tauMax, false/*warnIfAbsent*/);
}

void HyPerLCALayer::ioParam_numWindowX(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "numWindowX", &numWindowX, numWindowX);
   if(numWindowX != 1) {
      parent->ioParamValue(ioFlag, name, "windowSymX", &windowSymX, windowSymX);
   }
}

void HyPerLCALayer::ioParam_numWindowY(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "numWindowY", &numWindowY, numWindowY);
   if(numWindowY != 1) {
      parent->ioParamValue(ioFlag, name, "windowSymY", &windowSymY, windowSymY);
   }
}

void HyPerLCALayer::ioParam_windowSymX(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "numWindowX"));
}

void HyPerLCALayer::ioParam_windowSymY(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "numWindowY"));
}

void HyPerLCALayer::ioParam_selfInteract(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "selfInteract", &selfInteract, selfInteract);
}

void HyPerLCALayer::ioParam_slopeErrorStd(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "timeConstantTau"));
   assert(!parent->parameters()->presentAndNotBeenRead(name, "timeConstantTauMinimum"));
   if ((tauMax - tauMin) > 1.0) {
      parent->ioParamValue(ioFlag, name, "slopeErrorStd", &slopeErrorStd, slopeErrorStd, true/*warnIfAbsent*/);
   }
}

int HyPerLCALayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
      unsigned int * active_indices, unsigned int * num_active)
{
   update_timer->start();
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      updateStateOpenCL(time, dt);
      //HyPerLayer::updateState(time, dt);
   }
   else {
#endif
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int num_neurons = nx*ny*nf;
      dtTau = dt;
      double error_mean = 0;
      //Only update when the probe updates
      //TODO put following VThresh update into kernel
      if(dVThresh){
         if(fabs(sparseProbe->getUpdateTime() - (time)) <= (dt/2) && time > parent->getStartTime()+sparseProbe->getDeltaUpdateTime()){ 
            float actualSp = sparseProbe->getSparsity();
            //if (parent->columnId()==0) {
            //   std::cout << "Updating VThresh on time " << time << " from " << VThresh << " to ";
            //}
            VThresh += dVThresh*((actualSp - targetSparsity)/targetSparsity);
            //if(parent->columnId() == 0){
            //   std::cout << VThresh << " (target:" << targetSparsity << " actual:" << actualSp << ")\n";
            //}
         }
      }

      HyPerLCALayer_update_state(num_neurons, nx, ny, nf, loc->nb, numChannels,
            V, VThresh, AMax, AMin, AShift, VWidth, tauMax, tauMin, slopeErrorStd, 
            selfInteract, &dtTau, gSynHead, A, &error_mean, &errorStd);
      if (this->writeSparseActivity){
         updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      }
#ifdef PV_USE_OPENCL
   }
#endif

   update_timer->stop();
   return PV_SUCCESS;
}

} /* namespace PV */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/HyPerLCALayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/HyPerLCALayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif


