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
   if(xhistory){
      for (int j=0; j<bufferLength; j++) {
         free(xhistory[j]);
         xhistory[j] = NULL;
      }
      free(xhistory);
      xhistory = NULL;
   }
   if(timehistory){
      free(timehistory);
      timehistory = NULL;
   }
   if(targetFreqs){
      free(targetFreqs);
      targetFreqs = NULL;
   }
   if(deltaFreqs){
      free(deltaFreqs);
      deltaFreqs = NULL;
   }
   if(Mreal){
      free(Mreal);
      Mreal = NULL;
   }
   if(Mimag){
      free(Mimag);
      Mimag = NULL;
   }
}

int inverseCochlearLayer::initialize_base() {
   sampleRate = 0;
   inputLayername = NULL;
   cochlearLayername = NULL;
   inputLayer = NULL;
   cochlearLayer = NULL;
   xhistory = NULL;
   timehistory = NULL;
   targetFreqs = NULL;
   deltaFreqs = NULL;
   Mreal = NULL;
   Mimag = NULL;
   return PV_SUCCESS;
}

int inverseCochlearLayer::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);
   //Initialize any other member variables here
    nextDisplayTime = hc->getStartTime();

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
   
   numFrequencies = inputLayer->getLayerLoc()->nf;
   xhistory = (pvdata_t **) calloc(bufferLength, sizeof(pvdata_t *));
   assert(xhistory!=NULL); // TODO: change to error message
   for (int j=0; j<bufferLength; j++) {
      xhistory[j] = (pvdata_t *) calloc(numFrequencies, sizeof(pvdata_t));
      assert(xhistory[j] != NULL); // TODO: stop being lazy
   }
   ringBufferLevel = 0;
   
   timehistory = (double *) calloc(bufferLength, sizeof(double));
   if (timehistory==NULL) {
      fprintf(stderr, "Unable to allocate memory for timehistory: %s\n", strerror(errno));
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

   Mreal = (float **) calloc(bufferLength, sizeof(float *));
   Mimag = (float **) calloc(bufferLength, sizeof(float *));
   if (Mreal==NULL || Mimag==NULL) {
      fprintf(stderr, "Unable to allocate memory for Mreal or Mimag: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
   }
   double sampleFrequency = 1.0/cochlearLayer->getSampleRate();
   for (int j=0; j<bufferLength; j++) {
      Mreal[j] = (pvdata_t *) calloc(numFrequencies, sizeof(pvdata_t));
      Mimag[j] = (pvdata_t *) calloc(numFrequencies, sizeof(pvdata_t));
      if (Mreal[j]==NULL || Mimag[j]==NULL) {
         fprintf(stderr, "Unable to allocate memory for Mreal[j] or Mimag[j]: %s\n", strerror(errno));
         exit(EXIT_FAILURE);
      }
      for(int k=0; k<numFrequencies; k++) {
         Mreal[j][k] = targetFreqs[k]*cochlearLayer->getDampingConstants()[k]*deltaFreqs[k]*sin(targetFreqs[k]*j*sampleFrequency);
         Mimag[j][k] = targetFreqs[k]*cochlearLayer->getDampingConstants()[k]*deltaFreqs[k]*cos(targetFreqs[k]*j*sampleFrequency);
      }
   }
   
   return PV_SUCCESS;
}

int inverseCochlearLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_inputLayername(ioFlag);
   ioParam_cochlearLayername(ioFlag);
   ioParam_sampleRate(ioFlag);
   ioParam_bufferLength(ioFlag);

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
    if (time >= nextDisplayTime) {
       nextDisplayTime += cochlearLayer->getDisplayPeriod();
       const PVLayerLoc * loc = getLayerLoc();
       int nx = loc->nx;
       int ny = loc->ny;
       int nf = loc->nf;

       //This layer must be 1x1x(INVERSECOCHLEARLAYER_NF)
       assert(nx == 1 && ny == 1 && nf == INVERSECOCHLEARLAYER_NF);
       int num_input_neurons = inputLayer->getNumNeurons();
       int num_output_neurons = getNumNeurons();
       //num_output_neurons should be only INVERSECOCHLEARLAYER_NF
       assert(num_output_neurons == INVERSECOCHLEARLAYER_NF);
       
       timehistory[ringBufferLevel] = time;
       for (int k=0; k<inputLayer->getLayerLoc()->nf; k++) {
          xhistory[ringBufferLevel][k] = inputLayer->getLayerData()[k];
       } // memcpy?
       
       double sumreal = 0.0;
       double sumimag = 0.0;
       for (int j=0; j<bufferLength; j++) {
          for (int k=0; k<numFrequencies; k++) {
             sumreal += Mreal[j][k]*xhistory[ringBuffer(j)][k];
             sumimag += Mimag[j][k]*xhistory[ringBuffer(j)][k];
          }
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
       // clayer->activity->data[0] *= 0.25; // With bufferLength 1, sound is reproduced well but at a higher amplitude
       // clayer->activity->data[1] *= 0.25; // This corrects the amplitude to approximately its original value
                                             // But I think the correction factor depends on frequency.  --pfs Jun 23, 2014
       
       ringBufferLevel++;
       if (ringBufferLevel == bufferLength) { ringBufferLevel = 0; }
    }
    
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

