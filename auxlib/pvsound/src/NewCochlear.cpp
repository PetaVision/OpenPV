//
//  NewCochlear.cpp
//  
//
//  Created by Brohit's Mac 5054128794 on 7/29/14.
//
//

#include "NewCochlear.h"
#include <iostream>

#include <stdio.h>

namespace PVsound {
    
    NewCochlearLayer::NewCochlearLayer() {
        initialize_base();
    }
    
    NewCochlearLayer::NewCochlearLayer(const char * name, PV::HyPerCol * hc) {
        initialize_base();
        initialize(name, hc);
    }  // end NewCochlearLayer::NewCochlearLayer(const char *, PV::HyPerCol *)
    
    NewCochlearLayer::~NewCochlearLayer() {
        targetFreqs.clear();
        radianFreqs.clear();
        omegas.clear();
        dampingConstants.clear();
        free(inputLayername);
        free(vVal);
        free(xVal);
        
        
        filename = NULL;
        delete fileHeader;
        if(soundBuf){
            free(soundBuf);
        }
        soundBuf = NULL;
        sf_close(fileStream);
    }
    
    int NewCochlearLayer::initialize_base() {
        freqMin = 20; // hertz
        freqMax = 20000; // hertz
        dampingConstant = 0;
        inputLayer = NULL;
        inputLayername = NULL;
        targetChannel = 0;
        sampleRate = 0;
        cochlearScale = 0;
        vVal = NULL;
        xVal = NULL;
        omega = 0;
        equalTemperedFlag = false;
        spectrographFlag = false;
        
        frameStart= 0;
        filename = NULL;
        soundData = NULL;
        
        return PV_SUCCESS;
        
        return PV_SUCCESS;
        samplePeriod = 0;
    }
    
    int NewCochlearLayer::initialize(const char * name, PV::HyPerCol * hc) {
        int status = PV::HyPerLayer::initialize(name, hc);
        
        
        
        
        
        assert(filename!=NULL);
        filename = strdup(filename);
        assert(filename != NULL);
        fileHeader = new SF_INFO();
        fileStream = sf_open(filename, SFM_READ, fileHeader);
        assert(fileStream != NULL);
        sampleRate = fileHeader->samplerate;
        
        samplePeriod = 1.0/sampleRate; //hc->getDeltaTime();
        
        //Calculate nx
        targetFreqs.clear();
        targetFreqs.push_back(freqMin);
        radianFreqs.clear();
        radianFreqs.push_back(freqMin * 2 * PI);
        
        int nx = getLayerLoc()->nx;
        
        float newFreq = 0;
        float newradFreq = 0;
        
        if (!spectrographFlag)  {
        
        
            if (!equalTemperedFlag) {
                
                for(int i = 1; i < nx; i++){
                    float prevFreq = targetFreqs.back();
                    newFreq = 7e-10*powf(prevFreq,3) - 3e-6*powf(prevFreq,2) + 1.0041 * prevFreq + .6935;
                    //newFreq = prevFreq * powf(2,1/12.0); //for equal temperament
                    newradFreq = newFreq * 2 * PI;
                    targetFreqs.push_back(newFreq);
                    radianFreqs.push_back(newradFreq);
                    std::cout << ":: Frequency " << newFreq << "\n";
                }
            }
            
            else {
                for(int i = 1; i < nx; i++){
                    float prevFreq = targetFreqs.back();
                    //newFreq = 7e-10*powf(prevFreq,3) - 3e-6*powf(prevFreq,2) + 1.0041 * prevFreq + .6935;
                    newFreq = prevFreq * powf(2,1/12.0); //for equal temperament
                    newradFreq = newFreq * 2 * PI;
                    targetFreqs.push_back(newFreq);
                    radianFreqs.push_back(newradFreq);
                    std::cout << ":: Frequency " << newFreq << "\n";
                }

            }
            
        }
        
        else {
            
            for(int i = 1; i < nx; i++){
                float prevFreq = targetFreqs.back();
                float deltaFreq = (log(freqMax) - log(freqMin)) / nx;
                newFreq = prevFreq * exp(deltaFreq); //for spectrograph
                newradFreq = newFreq * 2 * PI;
                targetFreqs.push_back(newFreq);
                radianFreqs.push_back(newradFreq);
                std::cout << ":: Frequency " << newFreq << "\n";
            }
            
        }
        
        
        
        //This is not read from parameters, but set explicitly
        nx = targetFreqs.size();
       // std::cout << ":: nx " << nx << "\n";
        
        
        //This should have been set correctly
        assert(targetFreqs.size() > 0);
        assert(getLayerLoc()->nx == targetFreqs.size());
        
        //Set up damping constant based on frequency envelope
        dampingConstants.clear();
        omegas.clear();
        cochlearScales.clear();
        
        for(int i = 0; i < radianFreqs.size(); i++){
            
            
            
            
            dampingConstant = radianFreqs[i] / ( 12.7 * pow((radianFreqs[i] / 1000), .3)) ;
            
            
            omega = (.5 * sqrt( (4 * pow(radianFreqs[i], 2)) - pow(dampingConstant, 2)));
            
            cochlearScale = 2 * PI * radianFreqs[i] * dampingConstant;
          
            
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
    
    
    double NewCochlearLayer::getDeltaUpdateTime(){
        return parent->getDeltaTime(); // gets dt
    }
    
    int NewCochlearLayer::communicateInitInfo(){
        PV::HyPerLayer::communicateInitInfo();
        return PV_SUCCESS;
    }
    
    int NewCochlearLayer::allocateDataStructures(){
        PV::HyPerLayer::allocateDataStructures();
        
      //  free(clayer->V);
      //  clayer->V = NULL;
        
        // Point to clayer data struct
        soundData = clayer->activity->data;
        assert(soundData!=NULL);
        
        soundBuf = (float*) malloc(sizeof(float));
        if(frameStart <= 1){
            frameStart = 1;
        }
        //Set to frameStart, which is 1 indexed
        sf_seek(fileStream, frameStart-1, SEEK_SET);
        //status = updateState(0,parent->getDeltaTime());
        //Reset filepointer to reread the same frame on the 0th timestep
        sf_seek(fileStream, frameStart-1, SEEK_SET);
        return PV_SUCCESS;
    }
    
    int NewCochlearLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
        //This needs to be called here to grab max/min first
        ioParam_FreqMinMax(ioFlag);
        int status = PV::HyPerLayer::ioParamsFillGroup(ioFlag);
        ioParam_targetChannel(ioFlag);
        //ioParam_inputLayername(ioFlag);
        ioParam_sampleRate(ioFlag);
        ioParam_equalTemperedFlag(ioFlag);
        ioParam_spectrographFlag(ioFlag);
        
        ioParam_soundInputPath(ioFlag);
        ioParam_frameStart(ioFlag);
        
        return status;
    }
    
    void NewCochlearLayer::ioParam_nf(enum ParamsIOFlag ioFlag){
        assert(!parent->parameters()->presentAndNotBeenRead(name,"freqMin"));
        assert(!parent->parameters()->presentAndNotBeenRead(name,"freqMax"));
        
    }
    
    void NewCochlearLayer::ioParam_FreqMinMax(enum ParamsIOFlag ioFlag) {
        //Defaults are range of piano keys
        parent->ioParamValue(ioFlag, name, "freqMin", &freqMin, freqMin);
        parent->ioParamValue(ioFlag, name, "freqMax", &freqMax, freqMax);
        //Check freq ranges
        if(freqMin >= freqMax){
            std::cout << "NewCochlearLayer:: Frequency min must be smaller than freqMax\n";
            exit(EXIT_FAILURE);
        }
    }
    
    void NewCochlearLayer::ioParam_targetChannel(enum ParamsIOFlag ioFlag) {
        //Defaults are range of piano keys
        parent->ioParamValue(ioFlag, name, "targetChannel", &targetChannel, targetChannel);
    }
    
    void NewCochlearLayer::ioParam_dampingConstant(enum ParamsIOFlag ioFlag) {
        parent->ioParamValue(ioFlag, name, "dampingConstant", &dampingConstant, dampingConstant);
    }
    
    void NewCochlearLayer::ioParam_sampleRate(enum ParamsIOFlag ioFlag) {
        parent->ioParamValueRequired(ioFlag, name, "sampleRate", &sampleRate);
    }
    

    void NewCochlearLayer::ioParam_equalTemperedFlag(enum ParamsIOFlag ioFlag) {
        parent->ioParamValueRequired(ioFlag, name, "equalTemperedFlag", &equalTemperedFlag);
    }

    void NewCochlearLayer::ioParam_spectrographFlag(enum ParamsIOFlag ioFlag) {
        parent->ioParamValueRequired(ioFlag, name, "spectrographFlag", &spectrographFlag);
    }
    
    void NewCochlearLayer::ioParam_soundInputPath(enum ParamsIOFlag ioFlag) {
        parent->ioParamString(ioFlag, name, "soundInputPath", &filename, NULL, false/*warnIfAbsent*/);
    }
    
    void NewCochlearLayer::ioParam_frameStart(enum ParamsIOFlag ioFlag) {
        parent->ioParamValue(ioFlag, name, "frameStart", &frameStart, 0/*default value*/);
    }
    
    
    
    int NewCochlearLayer::updateState(double time, double dt){
        //update_timer->start();
        
        float numSamples = nearbyint(dt / samplePeriod); //assumes dt is an integer multiple of samplePeriod
        
      
        
        const PVLayerLoc * loc = getLayerLoc();
        int nx = loc->nx;
        int ny = loc->ny;
        int nf = loc->nf;
        assert(nf == 1 && ny == 1);
       // int num_input_neurons = inputLayer->getNumNeurons();
        int num_output_neurons = getNumNeurons();
        
        assert(fileStream);
        
        
        
        
        
    for (int i = 0; i < numSamples; i++) {
            
        //Read 1 frame
        int numRead = sf_readf_float(fileStream, soundBuf, 1);
        //EOF
        if(numRead == 0){
            sf_seek(fileStream, 0, SEEK_SET);
            numRead = sf_readf_float(fileStream, soundBuf, 1);
            if(numRead == 0){
                fprintf(stderr, "SoundStream:: Fatal error, is the file empty?\n");
                exit(EXIT_FAILURE);
            }
            std::cout << "Rewinding sound file\n";
        }
        else if(numRead > 1){
            fprintf(stderr, "SoundStream:: Fatal error, numRead is bigger than 1\n");
            exit(EXIT_FAILURE);
        }
        for(int fi = 0; fi < getLayerLoc()->nf; fi++){
            soundData[fi] = soundBuf[fi];
          //  std::cout << ":: sounddata " << soundData[fi] << "\n";
        }

        //Reset pointer of gSynHead to point to the excitatory channel
        //pvdata_t * inA = inputLayer->getCLayer()->activity->data;
        pvdata_t * V = getV();
        
        for(int inNi = 0; inNi < 1; inNi++){
            
            if(inNi == targetChannel){
                
                float inVal = soundData[inNi];
                //Loop through current layer's neurons
                
//#ifdef PV_USE_OPENMP_THREADS
//#pragma omp parallel for
//#endif
                
                for(int outNi = 0; outNi < num_output_neurons; outNi++){
                    
                    
                    
                    dampingConstant = dampingConstants[outNi];
                    
                    cochlearScale = cochlearScales[outNi];
                    
                    //std::cout << ":: radianfreqscochlea " << radianFreqs[outNi] << "\n";
                    
                    //float sound = sin (440 * time * 2 * PI);
                    
                    float c1 = xVal[outNi] - (inVal / pow(radianFreqs[outNi], 2));
                    float c2 = (vVal[outNi] + (.5 * dampingConstant) * c1) / omegas[outNi];
                    
                    float xtermone = c1 * exp(-.5 * dampingConstant * samplePeriod ) * cos(omegas[outNi] * samplePeriod );
                    float xtermtwo = c2 * exp(-.5 * dampingConstant * samplePeriod ) * sin(omegas[outNi] * samplePeriod );
                    
                    float vtermone = -.5 * dampingConstant * c1 * exp(-.5 * dampingConstant * samplePeriod ) * cos(omegas[outNi] * samplePeriod );
                    float vtermtwo = -1 * omegas[outNi] * c1 * exp(-.5 * dampingConstant * samplePeriod ) * sin(omegas[outNi] * samplePeriod );
                    float vtermthree = -.5 * dampingConstant * c2 * exp(-.5 * dampingConstant * samplePeriod ) * sin(omegas[outNi] * samplePeriod );
                    float vtermfour = omegas[outNi] * c2 * exp(-.5 * dampingConstant * samplePeriod ) * cos(omegas[outNi] * samplePeriod );
                    
                    xVal[outNi] = xtermone + xtermtwo + (inVal / pow(radianFreqs[outNi], 2));
                    
                    vVal[outNi] = vtermone + vtermtwo + vtermthree + vtermfour;
                    
                    //std::cout << ":: xVal " << xVal[outNi] << "\n";
                    
                    V[outNi] = xVal[outNi] * cochlearScale; //multiply by (non-freq dependent) inner ear amplification? (10,000,000 gets into reasonable range)
                    
                  //  std::cout << ":: Vbuffer " << V[outNi] << "\n";
                    
                }
            }
        }
            
    } // closes forloop over numSamples
        
    
    
        
        
    //Copy V to A buffer
    PV::HyPerLayer::setActivity();
        
        
        //update_timer->stop();
        return PV_SUCCESS;
        
    }
    
    PV::BaseObject * createNewCochlearLayer(char const * name, PV::HyPerCol * hc) {
        return hc ? new NewCochlearLayer(name, hc) : NULL;
    }
}  // end namespace PVsound
