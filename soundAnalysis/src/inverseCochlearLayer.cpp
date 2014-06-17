/*
 * inverseCochlearLayer.cpp
 *
 *  Created on: June 4, 2014
 *      Author: slundquist
 */

#include "inverseCochlearLayer.hpp"

namespace PV {

inverseCochlearLayer::inverseCochlearLayer() {
   initialize_base();
}

inverseCochlearLayer::inverseCochlearLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}  // end inverseCochlearLayer::inverseCochlearLayer(const char *, HyPerCol *)

inverseCochlearLayer::~inverseCochlearLayer() {
   if(inputLayername){
      free(inputLayername);
      inputLayername = NULL;
   }
   if(cochlearLayername){
      free(cochlearLayername);
      cochlearLayername = NULL;
   }
}

int inverseCochlearLayer::initialize_base() {
   sampleRate = 0;
   inputLayername = NULL;
   cochlearLayername = NULL;
   inputLayer = NULL;
   cochlearLayer = NULL;
   return PV_SUCCESS;
}

int inverseCochlearLayer::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);
   //Initialize any other variables here

   return status;
}

int inverseCochlearLayer::communicateInitInfo(){
   ANNLayer::communicateInitInfo();

   //Grab input layer stuff
   inputLayer = parent->getLayerFromName(inputLayername);
   if (inputLayer == NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: InputLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, inputLayername);
      }
      exit(EXIT_FAILURE);
   }
   //Make sure the size is correct for the input layer
   if(inputLayer->getLayerLoc()->nx != 1 || inputLayer->getLayerLoc()->ny != 1){
      fprintf(stderr, "%s \"%s\" error: InputLayer \"%s\" must have a nx and ny size of 1.\n",
              parent->parameters()->groupKeywordFromName(name), name, inputLayername);
      exit(EXIT_FAILURE);
   }

   //Grab the cochlear layer
   HyPerLayer* tempLayer = parent->getLayerFromName(cochlearLayername);
   if (tempLayer == NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: CochlearLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, cochlearLayername);
      }
      exit(EXIT_FAILURE);
   }

   cochlearLayer = dynamic_cast <CochlearLayer*> (tempLayer);
   if (cochlearLayer == NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: CochlearLayerName \"%s\" is not a CochlearLayer.\n",
                 parent->parameters()->groupKeywordFromName(name), name, cochlearLayername);
      }
      exit(EXIT_FAILURE);
   }

   return PV_SUCCESS;
}

int inverseCochlearLayer::allocateDataStructures(){
   ANNLayer::allocateDataStructures();

   return PV_SUCCESS;
}

int inverseCochlearLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_inputLayername(ioFlag);
   ioParam_cochlearLayername(ioFlag);
   ioParam_sampleRate(ioFlag);

   return status;
}

void inverseCochlearLayer::ioParam_sampleRate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "sampleRate", &sampleRate);
}

void inverseCochlearLayer::ioParam_inputLayername(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "inputLayername", &inputLayername);
}

void inverseCochlearLayer::ioParam_cochlearLayername(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "cochlearLayername", &cochlearLayername);
}

int inverseCochlearLayer::updateState(double time, double dt){
   update_timer->start();

   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;

   //This layer must be 1x1x1
   assert(nx == 1 && ny == 1 && nf == 1);
   int num_input_neurons = inputLayer->getNumNeurons();
   int num_output_neurons = getNumNeurons();
   //num_output_neurons should be only 1
   assert(num_output_neurons == 1);
    
   //Reset pointer of gSynHead to point to the excitatory channel
   pvdata_t * inA = inputLayer->getCLayer()->activity->data;
   pvdata_t * outV = getV();

   //*outV is where the output data should go

   //num_input_neurons is the spring number
   for(int inNi = 0; inNi < num_input_neurons; inNi++){
      //inVal is the displacement value of the input at spring num_input_neurons
      float inVal = inA[inNi];

   }

//Copy V to A buffer
   HyPerLayer::setActivity();
   update_timer->stop();
   return PV_SUCCESS;
}

}  // end namespace PV

