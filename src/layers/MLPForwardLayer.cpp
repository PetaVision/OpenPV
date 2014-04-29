
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
   if(randState) delete randState;
}

int MLPForwardLayer::initialize_base()
{
   randState = NULL;
   bias = NULL;
   initBiasType = NULL;
   dropoutChance = 0;
   normFactor = 1;
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

int MLPForwardLayer::allocateDataStructures() {
   int status = ANNLayer::allocateDataStructures();
   // // a random state variable is needed for every neuron/clthread
   randState = new Random(parent, getLayerLoc(), false/*isExtended*/);
   if (randState == NULL) {
      fprintf(stderr, "MLPForwardLayer::initialize error.  Layer \"%s\" unable to create object of Random class.\n", getName());
      exit(EXIT_FAILURE);
   }
   return status;
}

int MLPForwardLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_InitBiasType(ioFlag);
   ioParam_NormFactor(ioFlag);
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

void MLPForwardLayer::ioParam_DropoutChance(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dropoutChance", &dropoutChance, dropoutChance);
   if(dropoutChance < 0 || dropoutChance >= 1){
      fprintf(stderr, "%s \"%s\": dropoutChance must be between 0 (inclusive) and 1 (exclusive).\n",
            parent->parameters()->groupKeywordFromName(name), name);
   }
}

void MLPForwardLayer::ioParam_NormFactor(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "normFactor", &normFactor, normFactor);
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
      //Normalize values based on size
      //Dropout implemented here
      //Warning: this code assumes 2 things:
      //1. Given F as the activation function of MLP, F(-infinity) = 0;
      //2. Given F as the activation function of MLP, F'(-infinity) = 0;
      //If both f(V) and f'(V) are 0, both x and error of that node become 0, which prevents
      //updating W and B.
      double p = randState->uniformRandom();
      //In case uniform random returns 0
      if(p < dropoutChance || dropoutChance == 0){
         //Update V: GSynExt(channel 0) + bias
         V[ni] = GSynExt[ni] + bias[ni];
         V[ni] /= normFactor;
         //if(strcmp(name, "ForwardLayerFinal") == 0){
         //   std::cout << "time:" << time << "  ni:" << ni << "  Ext:" << GSynExt[ni] << "  bias:" << bias[ni] << "  V:" << V[ni] << "\n";
         //}
      }
      else{
         //Set to negative infinity
         //TODO no longer works with new sigmoid
         V[ni] = -999999999;
      }
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
   return status;
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
