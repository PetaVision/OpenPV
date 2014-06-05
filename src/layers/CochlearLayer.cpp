/*
 * CochlearLayer.cpp
 *
 *  Created on: June 4, 2014
 *      Author: slundquist
 */

#include "CochlearLayer.hpp"

namespace PV {

CochlearLayer::CochlearLayer() {
   initialize_base();
}

CochlearLayer::CochlearLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}  // end CochlearLayer::CochlearLayer(const char *, HyPerCol *)

CochlearLayer::~CochlearLayer() {
   targetFreqs.clear();
   dampingConstants.clear();
   free(inputLayername);
   free(vVal);
}

int CochlearLayer::initialize_base() {
   freqMin = 27.5;
   freqMax = 4186.01;
   dampingRatio = .5;
   inputLayer = NULL;
   inputLayername = NULL;
   targetChannel = 0;
   sampleRate = 0;
   vVal = NULL;
   return PV_SUCCESS;
}

int CochlearLayer::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);

   //This should have been set correctly
   assert(targetFreqs.size() > 0);
   assert(getLayerLoc()->nf == targetFreqs.size());

   //Set up damping constant based on the damping ratio
   dampingConstants.clear();
   for(int i = 0; i < targetFreqs.size(); i++){
      float constant = dampingRatio * 4 * PI * (targetFreqs[i]);
      dampingConstants.push_back(constant);
      std::cout << "damping constant: " << constant << "\n";
   }

   //Allocate buffers
   vVal = (float*) calloc(targetFreqs.size(), sizeof(float));
   assert(vVal);

   return status;
}

int CochlearLayer::communicateInitInfo(){
   ANNLayer::communicateInitInfo();
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
   //Make sure the nf for the input layer is correct
   //Only options are 1 for mono and 2 for stereo
   if(inputLayer->getLayerLoc()->nf != 1 && inputLayer->getLayerLoc()->nf != 2){
      fprintf(stderr, "%s \"%s\" error: InputLayer \"%s\" must have a nf size of 1 or 2.\n",
              parent->parameters()->groupKeywordFromName(name), name, inputLayername);
      exit(EXIT_FAILURE);
   }

   if(targetChannel > inputLayer->getLayerLoc()->nf){
      std::cout << "CochlearLayer:: InputLayer only has " << inputLayer->getLayerLoc()->nf << " channels, while target channel is set to " << targetChannel << "\n";
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int CochlearLayer::allocateDataStructures(){
   ANNLayer::allocateDataStructures();
   return PV_SUCCESS;
}

int CochlearLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   //This needs to be called here to grab max/min first
   ioParam_FreqMinMax(ioFlag);
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_targetChannel(ioFlag);
   ioParam_inputLayername(ioFlag);
   ioParam_sampleRate(ioFlag);
   return status;
}

void CochlearLayer::ioParam_nf(enum ParamsIOFlag ioFlag){
   assert(!parent->parameters()->presentAndNotBeenRead(name,"freqMin"));
   assert(!parent->parameters()->presentAndNotBeenRead(name,"freqMax"));

   if(ioFlag == PARAMS_IO_READ){
      //Calculate num features
      targetFreqs.clear();
      targetFreqs.push_back(freqMin);
      float newFreq = 0;
      while(newFreq <= freqMax){
         float prevFreq = targetFreqs.back();
         newFreq = 7e-10*powf(prevFreq,3) - 3e-6*powf(prevFreq,2) + 1.0041 * prevFreq + .6935;
         targetFreqs.push_back(newFreq);
      }
      //This is not read from parameters, but set explicitly
      numFeatures = targetFreqs.size();
      std::cout << "CochlearLayer " << name << ":: numFeatures set to " << numFeatures << "\n";
   }
}

void CochlearLayer::ioParam_FreqMinMax(enum ParamsIOFlag ioFlag) {
   //Defaults are range of piano keys
   parent->ioParamValue(ioFlag, name, "freqMin", &freqMin, freqMin);
   parent->ioParamValue(ioFlag, name, "freqMax", &freqMax, freqMax);
   //Check freq ranges
   if(freqMin >= freqMax){
      std::cout << "CochlearLayer:: Frequency min must be smaller than freqMax\n";
      exit(EXIT_FAILURE);
   }
}

void CochlearLayer::ioParam_targetChannel(enum ParamsIOFlag ioFlag) {
   //Defaults are range of piano keys
   parent->ioParamValue(ioFlag, name, "targetChannel", &targetChannel, targetChannel);
}

void CochlearLayer::ioParam_dampingRatio(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dampingRatio", &dampingRatio, dampingRatio);
}

void CochlearLayer::ioParam_sampleRate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "sampleRate", &sampleRate);
}

void CochlearLayer::ioParam_inputLayername(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "inputLayername", &inputLayername);
}

int CochlearLayer::updateState(double time, double dt){
   update_timer->start();

   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   assert(nx == 1 && ny == 1);
   int num_input_neurons = inputLayer->getNumNeurons();
   int num_output_neurons = getNumNeurons();
   //Reset pointer of gSynHead to point to the excitatory channel
   pvdata_t * inA = inputLayer->getCLayer()->activity->data;
   pvdata_t * V = getV();
   for(int inNi = 0; inNi < num_input_neurons; inNi++){
      int fi = featureIndex(inNi, nx, ny, nf);
      if(fi == targetChannel){
         float inVal = inA[inNi];
         //Loop through current layer's neurons
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for(int outNi = 0; outNi < num_output_neurons; outNi++){
            float kVal = (-4*PI*powf(targetFreqs[outNi], 2));
            float aVal = inVal - dampingConstants[outNi]*vVal[outNi] - kVal*V[outNi]; 
            //Update v and x values
            vVal[outNi] += aVal * sampleRate;
            //V buffer is the position
            V[outNi] += vVal[outNi] * sampleRate;

            ////Multiplying by k constant to get back out the value
            //float prevVal = V[outNi] * kVal;
            ////Accumulating outNi here
            //V[outNi] = inVal/kVal;
            ////Calculate damping
            //V[outNi] = V[outNi] - dampingConstants[outNi]*(inVal - prevVal);
            ////V[outNi] = V[outNi] - dampingRatio*(inVal - prevVal);
         }
      }
   }
   //Copy V to A buffer
   HyPerLayer::setActivity();
   update_timer->stop();
   return PV_SUCCESS;
}

}  // end namespace PV

