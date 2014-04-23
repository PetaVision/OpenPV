
/*
 * MLPForwardLayer.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist 
 */

#include "MLPForwardLayer.hpp"
#include "../include/default_params.h"

//#ifdef __cplusplus
//extern "C" {
//#endif
//
//void MLPForwardLayer_update_state(
//    const int numNeurons,
//    const int nx,
//    const int ny,
//    const int nf,
//    const int nb,
//
//    float * V,
//    const float Vth,
//    const float VMax,
//    const float VMin,
//    const float VShift,
//    float * GSynHead,
//    float * activity
//    );
//
//
//#ifdef __cplusplus
//}
//#endif

namespace PV {

MLPForwardLayer::MLPForwardLayer()
{
   initialize_base();
}

MLPForwardLayer::MLPForwardLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

MLPForwardLayer::~MLPForwardLayer()
{
   if(bias) free(bias);
}

int MLPForwardLayer::initialize_base()
{
   bias = NULL;
   initBiasType = NULL;
   return PV_SUCCESS;
}

int MLPForwardLayer::initialize(const char * name, HyPerCol * hc)
{
   int status = ANNLayer::initialize(name, hc);
   int numNeurons = getNumExtended();
   bias = (float*)calloc(numNeurons, sizeof(float));
   assert(initBiasType);
   if(strcmp(initBiasType, "File") == 0){
      assert(biasFilename);
      status |= readBias(biasFilename);
   }

   return status;
}

int MLPForwardLayer::communicateInitInfo(){
   int status = ANNLayer::communicateInitInfo();
   //Make sure all connections attached here do not have plasticity on
   return status;
}

int MLPForwardLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_InitBiasType(ioFlag);
   return status;
}

void MLPForwardLayer::ioParam_InitBiasType(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "InitBiasType", &initBiasType);
   //TODO more initial values, maybe convert V stuff code to do this part
   if(strcmp(initBiasType, "File") == 0){
      ioParam_BiasFilename(ioFlag);
   }
   else if(strcmp(initBiasType, "Zero") == 0){
      //Correct, do nothing
   }
   else{
      fprintf(stderr, "%s \"%s\": InitBiasType of \"%s\" not known, current options are \"Zero\" or \"File\".\n",
              parent->parameters()->groupKeywordFromName(name), name, initBiasType);
      exit(EXIT_FAILURE);
   }
}

void MLPForwardLayer::ioParam_BiasFilename(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "BiasFilename", &biasFilename);
}

int MLPForwardLayer::updateState(double time, double dt)
{
   update_timer->start();

   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   //Reset pointer of gSynHead to point to the excitatory channel
   pvdata_t * GSynExt = getChannel(CHANNEL_EXC);

   //Calculate constants for derivitive of sigmoid layer
   pvdata_t * V = getV();

   for(int ni = 0; ni < num_neurons; ni++){
      int next = kIndexExtended(ni, nx, ny, nf, loc->nb);
      //Update V: GSynExt(channel 0) + bias
      V[ni] = GSynExt[ni] + bias[ni];
   }
   //A never updated, TODO see if you can remove A buffer
   update_timer->stop();
   return PV_SUCCESS;
}

int MLPForwardLayer::readBias(const char * filename){
   InterColComm * icComm = parent->icCommunicator();
   double filetime;
   int status = readBufferFile(filename, icComm, &filetime, &bias, 1, /*extended*/false, getLayerLoc());
   assert(status == PV_SUCCESS);
   return status;
}

int MLPForwardLayer::checkpointRead(const char * cpDir, double * timef){
   int status = ANNLayer::checkpointRead(cpDir, timef);
   char basepath[PV_PATH_MAX];
   char filename[PV_PATH_MAX];
   int lenbase = snprintf(basepath, PV_PATH_MAX, "%s/%s", cpDir, name);
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s_B.pvp", basepath);
   status |= readBias(filename);
   assert(chars_needed < PV_PATH_MAX);
}

int MLPForwardLayer::checkpointWrite(const char * cpDir){
   int status = ANNLayer::checkpointWrite(cpDir);
   InterColComm * icComm = parent->icCommunicator();
   char basepath[PV_PATH_MAX];
   char filename[PV_PATH_MAX];
   int lenbase = snprintf(basepath, PV_PATH_MAX, "%s/%s", cpDir, name);
   double timed = (double) parent->simulationTime();

   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s_B.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   status |= writeBufferFile(filename, icComm, timed, &bias, 1, /*extended*/false, getLayerLoc());
   assert(status == PV_SUCCESS);
   return status;
}

} /* namespace PV */


//#ifdef __cplusplus
//extern "C" {
//#endif
//
//#ifndef PV_USE_OPENCL
//#  include "../kernels/MLPForwardLayer_update_state.cl"
//#else
//#  undef PV_USE_OPENCL
//#  include "../kernels/MLPForwardLayer_update_state.cl"
//#  define PV_USE_OPENCL
//#endif
//
//#ifdef __cplusplus
//}
//#endif
