
/*
 * MLPForwardLayer.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist 
 */

#include "MLPForwardLayer.hpp"
#include <include/default_params.h>

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

namespace PVMLearning {

MLPForwardLayer::MLPForwardLayer()
{
   initialize_base();
}

MLPForwardLayer::MLPForwardLayer(const char * name, PV::HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

MLPForwardLayer::~MLPForwardLayer()
{
   //if(bias) free(bias);
   if(dropout) free(dropout);
   if(randState) delete randState;
}

int MLPForwardLayer::initialize_base()
{
   randState = NULL;
   dropoutChance = 0;
   potentialScale = 1;
   return PV_SUCCESS;
}

int MLPForwardLayer::initialize(const char * name, PV::HyPerCol * hc)
{
   int status = ANNLayer::initialize(name, hc);
   int numNeurons = getNumExtendedAllBatches();
   //bias = (float*)calloc(numNeurons, sizeof(float));
   dropout = (bool*)calloc(numNeurons, sizeof(bool));
   //assert(initBiasType);
   //if(strcmp(initBiasType, "File") == 0){
   //   assert(biasFilename);
   //   status |= readBias(biasFilename);
   //}

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
   randState = new PV::Random(parent, getLayerLoc(), false/*isExtended*/);
   if (randState == NULL) {
      fprintf(stderr, "MLPForwardLayer::initialize error.  Layer \"%s\" unable to create object of Random class.\n", getName());
      exit(EXIT_FAILURE);
   }
   return status;
}

int MLPForwardLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_DropoutChance(ioFlag);
   ioParam_PotentialScale(ioFlag);
   return status;
}

void MLPForwardLayer::ioParam_DropoutChance(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dropoutChance", &dropoutChance, dropoutChance);
   if(dropoutChance < 0 || dropoutChance >= 1){
      fprintf(stderr, "%s \"%s\": dropoutChance must be between 0 (inclusive) and 1 (exclusive).\n",
            getKeyword(), name);
   }
}

void MLPForwardLayer::ioParam_PotentialScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "potentialScale", &potentialScale, potentialScale);
}

int MLPForwardLayer::updateState(double time, double dt)
{
   //update_timer->start();

   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   //Reset pointer of gSynHead to point to the excitatory channel
   pvdata_t * GSynExt = getChannel(CHANNEL_EXC);

   pvdata_t * V = getV();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for(int ni = 0; ni < num_neurons*loc->nbatch; ni++){
      //int next = kIndexExtended(ni, nx, ny, nf, loc->nb);
      double p = randState->uniformRandom();
      if(p <= dropoutChance){
         //Set dropout flag
         dropout[ni] = true;
      }
      else{
         dropout[ni] = false;
      }
      V[ni] = GSynExt[ni] * potentialScale;
      //if(strcmp(name, "ForwardLayerFinal") == 0){
      //   std::cout << "Neuron: " << ni << " V: " << V[ni] << "\n";
      //}
   }
   //A never updated, TODO see if you can remove A buffer
   //update_timer->stop();
   return PV_SUCCESS;
}

PV::BaseObject * createMLPForwardLayer(char const * name, PV::HyPerCol * hc) { 
   return hc ? new MLPForwardLayer(name, hc) : NULL;
}

} /* namespace PVMLearning */


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
