/*
 * inverseNewCochlearLayer.cpp
 *
 *  Created on: June 4, 2014
 *      Author: slundquist
 */

#include "inverseNewCochlearLayer.hpp"


inverseNewCochlearLayer::inverseNewCochlearLayer() {
    initialize_base();
}

inverseNewCochlearLayer::inverseNewCochlearLayer(const char * name, PV::HyPerCol * hc) {
    initialize_base();
    initialize(name, hc);
}  // end inverseNewCochlearLayer::inverseNewCochlearLayer(const char *, PV::HyPerCol *)

inverseNewCochlearLayer::~inverseNewCochlearLayer() {
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
    
    
    if(xVal){
        free(xVal);
        xVal = NULL;
    }
    
    if(lastxVal){
        free(lastxVal);
        lastxVal = NULL;
    }
    
    if(pastxVal){
        free(lastxVal);
        lastxVal = NULL;
    }
    
    if(vVal){
        free(vVal);
        vVal = NULL;
    }
    
    if(lastvVal){
        free(lastvVal);
        lastvVal = NULL;
    }
    
    if(sound){
        free(sound);
         sound = NULL;
    }
    
    if(energy){
        free(energy);
        energy = NULL;
    }
    
    if(omegas){
        free(omegas);
        omegas = NULL;
    }
    
    if(radianFreqs){
        free(radianFreqs);
        radianFreqs = NULL;
    }
    
    if(dampingConstants){
        free(dampingConstants);
        dampingConstants = NULL;
    }
}

int inverseNewCochlearLayer::initialize_base() {
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
    xVal = NULL;
    lastxVal = NULL;
    vVal = NULL;
    lastvVal = NULL;
    pastxVal = NULL;
    sound = NULL;
    energy = NULL;
    outputsound = 0;
    sumenergy = 0;
    return PV_SUCCESS;
}

int inverseNewCochlearLayer::initialize(const char * name, PV::HyPerCol * hc) {
    int status = ANNLayer::initialize(name, hc);
    //Initialize any other member variables here
    nextDisplayTime = hc->getStartTime();
    
    return status;
}

int inverseNewCochlearLayer::communicateInitInfo(){
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
    
    //Grab the cochlear layer
    PV::HyPerLayer* tempLayer = parent->getLayerFromName(cochlearLayername);
    if (tempLayer == NULL) {
        if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: CochlearLayerName \"%s\" is not a layer in the HyPerCol.\n",
                    parent->parameters()->groupKeywordFromName(name), name, cochlearLayername);
        }
        exit(EXIT_FAILURE);
    }
    
    cochlearLayer = dynamic_cast <PVsound::NewCochlearLayer*> (tempLayer); //this should be new cochlear layer
    if (cochlearLayer == NULL) {
        if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: CochlearLayerName \"%s\" is not a CochlearLayer.\n",
                    parent->parameters()->groupKeywordFromName(name), name, cochlearLayername);
        }
        exit(EXIT_FAILURE);
    }
    
    return PV_SUCCESS;
}

int inverseNewCochlearLayer::allocateDataStructures(){
    
    ANNLayer::allocateDataStructures();
    
    numFrequencies = cochlearLayer->getLayerLoc()->nx;
    
    omegas = (float *) calloc(numFrequencies, sizeof(float));
    radianFreqs = (float *) calloc(numFrequencies, sizeof(float));
    targetFreqs = (float *) calloc(numFrequencies, sizeof(float));
    dampingConstants = (float *) calloc(numFrequencies, sizeof(float));
    sound = (float *) calloc(numFrequencies, sizeof(float));
    energy = (float *) calloc(numFrequencies, sizeof(float));
    xVal = (float *) calloc(numFrequencies, sizeof(float));
    lastxVal = (float *) calloc(numFrequencies, sizeof(float));
    pastxVal = (float *) calloc(numFrequencies, sizeof(float));
    vVal = (float *) calloc(numFrequencies, sizeof(float));
    lastvVal = (float *) calloc(numFrequencies, sizeof(float));
    
    
    if (omegas==NULL || radianFreqs==NULL || dampingConstants==NULL) {
        fprintf(stderr, "Unable to allocate memory for omegas, radianfreqs, or dampingconstants: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    
    for (int k=0; k<numFrequencies; k++) {
        omegas[k] = cochlearLayer->getOmegas()[k];
        radianFreqs[k] = cochlearLayer->getRadianFreqs()[k];
        dampingConstants[k] = cochlearLayer->getDampingConstants()[k];
        targetFreqs[k] = cochlearLayer->getTargetFreqs()[k];
    }

    return PV_SUCCESS;
}


int inverseNewCochlearLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
    int status = ANNLayer::ioParamsFillGroup(ioFlag);
    ioParam_inputLayername(ioFlag);
    ioParam_cochlearLayername(ioFlag);
    ioParam_sampleRate(ioFlag);
    ioParam_bufferLength(ioFlag);
    
    return status;
}


void inverseNewCochlearLayer::ioParam_sampleRate(enum ParamsIOFlag ioFlag) {
    parent->ioParamValueRequired(ioFlag, name, "sampleRate", &sampleRate);
}

void inverseNewCochlearLayer::ioParam_bufferLength(enum ParamsIOFlag ioFlag) {
    parent->ioParamValueRequired(ioFlag, name, "bufferLength", &bufferLength);
}

void inverseNewCochlearLayer::ioParam_inputLayername(enum ParamsIOFlag ioFlag) {
    parent->ioParamStringRequired(ioFlag, name, "inputLayername", &inputLayername);
}

void inverseNewCochlearLayer::ioParam_cochlearLayername(enum ParamsIOFlag ioFlag) {
    parent->ioParamStringRequired(ioFlag, name, "cochlearLayername", &cochlearLayername);
}


int inverseNewCochlearLayer::updateState(double time, double dt){
    
    update_timer->start();
    
  
    
    const PVLayerLoc * loc = getLayerLoc();
    int nx = loc->nx;
    int ny = loc->ny;
    int nf = loc->nf;
    
    
    
  /*  for (int k=0; k<cochlearLayer->getLayerLoc()->nx; k++) {
        
        xVal[k] = inputLayer->getLayerData()[k];
        
        float sinterm = exp( -0.5 * dampingConstants[k] * dt) * sin(omegas[k] * dt);
        float costerm = exp( -0.5 * dampingConstants[k] * dt) * cos(omegas[k] * dt);
        
        float numerator = (lastxVal[k] * costerm) + (((lastvVal[k] + (.5 * dampingConstants[k] * lastxVal[k])) / omegas[k]) * sinterm) - xVal[k];
        float denominator = costerm + ((dampingConstants[k] / (2 * omegas[k])) * sinterm) - 1;
        
        sound[k] = pow(radianFreqs[k],2) * (numerator / denominator);
        
        float c1 = lastxVal[k] - (sound[k] / pow(radianFreqs[k],2));
        float c2 = (lastvVal[k] + (0.5 * dampingConstants[k] * c1)) / omegas[k];
        
        lastvVal[k] = vVal[k];
        
        vVal[k] = (-0.5 * dampingConstants[k] * c1 * costerm) - (omegas[k] * c1 * sinterm) - (0.5 * c2 * sinterm) + (omegas[k] * c2 * costerm);
        
        lastxVal[k] = xVal[k];
        
        outputsound += sound[k];
        
        
    } */
    
    for (int k=0; k<cochlearLayer->getLayerLoc()->nx; k++) {
        
        xVal[k] = inputLayer->getLayerData()[k] / cochlearLayer->getCochlearScales()[k];
        vVal[k] = (xVal[k] - lastxVal[k]) / dt ;
        //float aVal = (vVal[k] - lastvVal[k]) / dt;
        float aVal = (xVal[k] - (2 * lastxVal[k]) + pastxVal[k]) / powf(dt,2);

        sound[k] = aVal + (dampingConstants[k] * vVal[k]) + (powf(radianFreqs[k],2) * xVal[k]);
        
        
        energy[k] = (0.5 * powf(vVal[k],2)) + (0.5 * powf(radianFreqs[k],2) * powf(xVal[k],2));
        
        sumenergy += energy[k];
        
        //outputsound += energy[k] * sound[k];
        
        outputsound += sound[k];
        
        
        pastxVal[k] = lastxVal[k];
        lastvVal[k] = vVal[k];
        lastxVal[k] = xVal[k];
        
        
 
    }
    
    pvdata_t * V = getV();
    
    //V[0] = outputsound / sumenergy;
    
    V[0] = outputsound / cochlearLayer->getLayerLoc()->nx;
    
    PV::HyPerLayer::setActivity();
    
    update_timer->stop();
    return PV_SUCCESS;
}
 

PV::BaseObject * create_inverseNewCochlearLayer(char const * name, PV::HyPerCol * hc) {
   return hc ? new inverseNewCochlearLayer(name, hc) : NULL;
}
