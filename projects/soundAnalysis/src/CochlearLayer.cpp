/*
 * CochlearLayer.cpp
 * Users/JEC/Desktop/newvision/sandbox/soundAnalysis/src/CochlearLayer.hpp
 *  Created on: June 4, 2014
 *      Author: slundquist
 */

#include "CochlearLayer.hpp"

CochlearLayer::CochlearLayer() {
   initialize_base();
}

CochlearLayer::CochlearLayer(const char * name, PV::HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}  // end CochlearLayer::CochlearLayer(const char *, PV::HyPerCol *)

CochlearLayer::~CochlearLayer() {
   targetFreqs.clear();
   radianFreqs.clear();
   omegas.clear();
   dampingConstants.clear();
   free(inputLayername);
   free(vVal);
   free(xVal);
   
}

int CochlearLayer::initialize_base() {
   freqMin = 440; // hertz
   freqMax = 4400; // hertz
   dampingConstant = 0;
   inputLayer = NULL;
   inputLayername = NULL;
   targetChannel = 0;
   sampleRate = 0;
    cochlearScale = 0;
   vVal = NULL;
    xVal = NULL;
    omega = 0;
   return PV_SUCCESS;
    timestep = 0;
}

int CochlearLayer::initialize(const char * name, PV::HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);

    nextDisplayTime = hc->getStartTime();
    
    timestep = hc->getDeltaTime(); // 1.0/sampleRate;
    
    //Calculate nx
    targetFreqs.clear();
    targetFreqs.push_back(freqMin);
    radianFreqs.clear();
    radianFreqs.push_back(freqMin * 2 * PI);
    
    int nx = getLayerLoc()->nx;
    
    float newFreq = 0;
    float newradFreq = 0;
    
    for(int i = 1; i < nxScale; i++){
        float prevFreq = targetFreqs.back();
        newFreq = 7e-10*powf(prevFreq,3) - 3e-6*powf(prevFreq,2) + 1.0041 * prevFreq + .6935;
        //newFreq = prevFreq * powf(2,1/12.0); //for equal temperament
        newradFreq = newFreq * 2 * PI;
        targetFreqs.push_back(newFreq);
        radianFreqs.push_back(newradFreq);
        
    }
    
    
    //This is not read from parameters, but set explicitly
    nx = targetFreqs.size();
    std::cout << ":: nx " << nx << "\n";

    
   //This should have been set correctly
   assert(targetFreqs.size() > 0);
   assert(getLayerLoc()->nx == targetFreqs.size());

   //Set up damping constant based on frequency envelope
   dampingConstants.clear();
    omegas.clear();
    cochlearScales.clear();
    
    
   for(int i = 0; i < targetFreqs.size(); i++){
    
       
       
       
       dampingConstant = radianFreqs[i] / ( 12.7 * pow((radianFreqs[i] / 1000), .3)) ;
       
     
       omega = (.5 * sqrt( (4 * pow(radianFreqs[i], 2)) - pow(dampingConstant, 2)));
       
       cochlearScale = 2 * PI * radianFreqs[i] * dampingConstant;
      
       std::cout << "Freqs: " << targetFreqs[i] << "\n";
       
      dampingConstants.push_back(dampingConstant);
       omegas.push_back(omega);
       cochlearScales.push_back(cochlearScale);
   }

    
   //Allocate buffers
   vVal = (float*) calloc(radianFreqs.size(), sizeof(float));
   xVal = (float*) calloc(radianFreqs.size(), sizeof(float));
   assert(vVal);
   assert(xVal);
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
    int i;
    for (i=0;i<radianFreqs.size();i++)
    { std::cout << "radianFreqs: " << radianFreqs[i] << "\n";
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
    ioParam_cochlearScale(ioFlag);
    ioParam_displayPeriod(ioFlag);
   return status;
}

void CochlearLayer::ioParam_nf(enum ParamsIOFlag ioFlag){
   assert(!parent->parameters()->presentAndNotBeenRead(name,"freqMin"));
   assert(!parent->parameters()->presentAndNotBeenRead(name,"freqMax"));

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

void CochlearLayer::ioParam_dampingConstant(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dampingConstant", &dampingConstant, dampingConstant);
}

void CochlearLayer::ioParam_sampleRate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "sampleRate", &sampleRate);
}
    
void CochlearLayer::ioParam_cochlearScale(enum ParamsIOFlag ioFlag) {
    parent->ioParamValueRequired(ioFlag, name, "cochlearScale", &cochlearScale);
}

void CochlearLayer::ioParam_inputLayername(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "inputLayername", &inputLayername);
}
    
void CochlearLayer::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
    parent->ioParamValueRequired(ioFlag, name, "displayPeriod", &displayPeriod);
}

    
int CochlearLayer::updateState(double time, double dt){
   update_timer->start();

   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   assert(nf == 1 && ny == 1);
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
//#ifdef PV_USE_OPENMP_THREADS
//#pragma omp parallel for
//#endif
         for(int outNi = 0; outNi < num_output_neurons; outNi++){

           
         
             dampingConstant = dampingConstants[outNi];
             cochlearScale = cochlearScales[outNi];
             
             //float sound = sin (440 * time * 2 * PI);
           
             float c1 = xVal[outNi] - (inVal / pow(radianFreqs[outNi], 2));
             float c2 = (vVal[outNi] + (.5 * dampingConstant) * c1) / omegas[outNi];
             
             float xtermone = c1 * exp(-.5 * dampingConstant * timestep ) * cos(omegas[outNi] * timestep );
             float xtermtwo = c2 * exp(-.5 * dampingConstant * timestep ) * sin(omegas[outNi] * timestep );
             
             float vtermone = -.5 * dampingConstant * c1 * exp(-.5 * dampingConstant * timestep ) * cos(omegas[outNi] * timestep );
             float vtermtwo = -1 * omegas[outNi] * c1 * exp(-.5 * dampingConstant * timestep ) * sin(omegas[outNi] * timestep );
             float vtermthree = -.5 * dampingConstant * c2 * exp(-.5 * dampingConstant * timestep ) * sin(omegas[outNi] * timestep );
             float vtermfour = omegas[outNi] * c2 * exp(-.5 * dampingConstant * timestep ) * cos(omegas[outNi] * timestep );
             
             xVal[outNi] = xtermone + xtermtwo + (inVal / pow(radianFreqs[outNi], 2));
             
             vVal[outNi] = vtermone + vtermtwo + vtermthree + vtermfour;
             
            // std::cout << ":: xVal " << xVal[124] << "\n";
             
             V[outNi] = xVal[outNi] * cochlearScale; //multiply by (non-freq dependent) inner ear amplification? (100000 gets into reasonable range)
             
           // std::cout << ":: Vbuffer " << V[124] << "\n";
             
         }
      }
   }
    if (time >= nextDisplayTime){
        nextDisplayTime += displayPeriod;
        //Copy V to A buffer
        PV::HyPerLayer::setActivity();
    }
    
   update_timer->stop();
   return PV_SUCCESS;
    
}

PV::BaseObject * createCochlearLayer(char const * name, PV::HyPerCol * hc) {
   return hc ? new CochlearLayer(name, hc) : NULL;
}
