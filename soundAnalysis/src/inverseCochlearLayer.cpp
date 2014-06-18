/*
 * inverseCochlearLayer.cpp
 *
 *  Created on: June 4, 2014
 *      Author: slundquist
 */

#include "inverseCochlearLayer.hpp"

#define INVERSECOCHLEARLAYER_NF 2
// Two features, one for real part, one for imaginary part
// imaginary part should be negligible; change to one feature when that happens

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
   inputRingBuffer = NULL;
   timehistory = NULL;
   dthistory = NULL;
   targetFreqs = NULL;
   deltaFreqs = NULL;
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
   
   int numFrequencies = inputLayer->getLayerLoc()->nf;
   inputRingBuffer = (pvdata_t **) calloc(bufferLength, sizeof(pvdata_t));
   assert(inputRingBuffer!=NULL); // TODO: change to error message
   for (int t=0; t<bufferLength; t++) {
      inputRingBuffer[t] = (pvdata_t *) calloc(numFrequencies, sizeof(pvdata_t));
      assert(inputRingBuffer[t] != NULL); // TODO: stop being lazy
   }
   ringBufferLevel = 0;
   
   timehistory = (double *) calloc(bufferLength, sizeof(double));
   dthistory = (double *) calloc(bufferLength, sizeof(double));
   if (timehistory==NULL || dthistory==NULL) {
      fprintf(stderr, "Unable to allocate memory for timehistory or dthistory: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
   }
   
   targetFreqs = (float *) calloc(numFrequencies, sizeof(float));
   deltaFreqs = (float *) calloc(numFrequencies, sizeof(float));
   if (targetFreqs==NULL || deltaFreqs==NULL) {
      fprintf(stderr, "Unable to allocate memory for targetFreqs or deltaFreqs: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
   }
   for (int k=0; k<numFrequencies; k++) {
      targetFreqs[k] = cochlearLayer->getTargetFreqs()[k];
   }
   for (int k=0; k<numFrequencies-1; k++) {
      deltaFreqs[k] = targetFreqs[k+1]-targetFreqs[k];
   }
   float lastfreq = targetFreqs[numFrequencies-1];
   float nextfreq = 7e-10*powf(lastfreq,3) - 3e-6*powf(lastfreq,2) + 1.0041*lastfreq+0.6935;
   deltaFreqs[numFrequencies-1] = nextfreq-lastfreq;

   return PV_SUCCESS;
}

int inverseCochlearLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_inputLayername(ioFlag);
   ioParam_cochlearLayername(ioFlag);
   ioParam_sampleRate(ioFlag);

   return status;
}

void inverseCochlearLayer::ioParam_nf(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      numFeatures = INVERSECOCHLEARLAYER_NF;
      parent->parameters()->handleUnnecessaryParameter(name, "nf", INVERSECOCHLEARLAYER_NF);
   }
}

void inverseCochlearLayer::ioParam_sampleRate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "sampleRate", &sampleRate);
}

void inverseCochlearLayer::ioParam_bufferLength(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "bufferLength", &bufferLength);
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
   assert(nx == 1 && ny == 1 && nf == INVERSECOCHLEARLAYER_NF);
   int num_input_neurons = inputLayer->getNumNeurons();
   int num_output_neurons = getNumNeurons();
   //num_output_neurons should be only INVERSECOCHLEARLAYER_NF
   assert(num_output_neurons == INVERSECOCHLEARLAYER_NF);
   
   timehistory[ringBufferLevel] = time;
   dthistory[ringBufferLevel] = dt;
   for (int k=0; k<inputLayer->getLayerLoc()->nf; k++) {
      inputRingBuffer[ringBufferLevel][k] = inputLayer->getLayerData()[k];
   } // memcpy?
   
   double sumreal = 0.0;
   double sumimag = 0.0;
   for (int k=0; k<inputLayer->getLayerLoc()->nf; k++) {
      float freq = targetFreqs[k];
      double innersumreal = 0.0;
      double innersumimag = 0.0;
      for (int j=0; j<bufferLength; j++) {
         float x = inputRingBuffer[ringBuffer(j)][k];
         double theta = freq*(time-timehistory[ringBuffer(j)]);
         innersumreal += sin(theta)*x*dthistory[ringBuffer(j)];
         innersumimag += cos(theta)*x*dthistory[ringBuffer(j)];
      }
      float kfactor = cochlearLayer->getDampingConstants()[k] * targetFreqs[k] * deltaFreqs[k];
      innersumreal *= kfactor;
      innersumimag *= kfactor;
      sumreal += innersumreal;
      sumimag += innersumimag;
   }
   sumreal /= (2*PI);
   sumimag /= (2*PI);
   
   
   //Reset pointer of gSynHead to point to the excitatory channel
   // pvdata_t * inA = inputLayer->getCLayer()->activity->data;
   pvdata_t * outV = getV();
   
   outV[0] = sumreal;
   outV[1] = sumimag;

   //*outV is where the output data should go

//Copy V to A buffer
   HyPerLayer::setActivity();
   
   
   ringBufferLevel++;
   if (ringBufferLevel == bufferLength) { ringBufferLevel = 0; }
   
   update_timer->stop();
   return PV_SUCCESS;
}

int inverseCochlearLayer::ringBuffer(int level) {
   int b = ringBufferLevel - level;
   b = b % bufferLength;
   if (b<0) {
      b += bufferLength;
   }
   assert(b>=0 && b<bufferLength);
   return b;
}

}  // end namespace PV

