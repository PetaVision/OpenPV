/*
 * SoundStream.cpp
 *
 *  Created on: May 6, 2013
 *      Author: slundquist 
 */

#include "SoundStream.hpp"
#include <iostream>

#include <stdio.h>

namespace PVsound {

SoundStream::SoundStream(){
   initialize_base();
}

SoundStream::SoundStream(const char * name, PV::HyPerCol * hc){
   initialize_base();
   initialize(name, hc);
}

SoundStream::~SoundStream() {
   filename = NULL;
   delete fileHeader;
   if(soundBuf){
      free(soundBuf);
   }
   soundBuf = NULL;
   sf_close(fileStream);
   //Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
   //if (getParent()->icCommunicator()->commRank()==0 && fileStream != NULL && fileStream->isfile) {
   //  PV_fclose(fileStream);
   //}
}

int SoundStream::initialize_base() {
   // displayPeriod = 1;
   frameStart= 0;
   filename = NULL;
   soundData = NULL;

   return PV_SUCCESS;
}

int SoundStream::initialize(const char * name, PV::HyPerCol * hc) {
   int status = PV_SUCCESS;
   PV::HyPerLayer::initialize(name, hc);

   //Only one mpi process allowed
   assert(getParent()->icCommunicator()->commSize() == 1);

   assert(filename!=NULL);
   filename = strdup(filename);
   assert(filename != NULL);
   fileHeader = new SF_INFO();
   fileStream = sf_open(filename, SFM_READ, fileHeader);
   assert(fileStream != NULL);
    sampleRate = fileHeader->samplerate;
   // nextSampleTime = hc->getStartTime();
   return status;
}



double SoundStream::getDeltaUpdateTime(){
   return 1.0/sampleRate; 
}

int SoundStream::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   PV::HyPerLayer::ioParamsFillGroup(ioFlag);

   ioParam_soundInputPath(ioFlag);
   ioParam_frameStart(ioFlag);
   return PV_SUCCESS;
}

void SoundStream::ioParam_soundInputPath(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "soundInputPath", &filename, NULL, false/*warnIfAbsent*/);
}

void SoundStream::ioParam_frameStart(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "frameStart", &frameStart, 0/*default value*/);
}

void SoundStream::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   //No need for a V
   initVObject = NULL;
}

int SoundStream::setActivity(){
   //Does nothing
   return PV_SUCCESS;
}


int SoundStream::allocateDataStructures() {
   int status = PV::HyPerLayer::allocateDataStructures();
   free(clayer->V);
   clayer->V = NULL;

   // Point to clayer data struct
   soundData = clayer->activity->data;
   assert(soundData!=NULL);
   //Layer must be 1 by 1 by 1
   if(getLayerLoc()->nx != 1 || getLayerLoc()->ny != 1){
      fprintf(stderr, "SoundStream::SoundStream layer must be 1 by 1 in the x and y direction\n");
      exit(EXIT_FAILURE);
   }
   if(getLayerLoc()->nf > fileHeader->channels){
      fprintf(stderr, "SoundStream::Audio file has %d channels, while the number of features is %d\n", fileHeader->channels, getLayerLoc()->nf);
      exit(EXIT_FAILURE);
   }
   //Allocate read buffer based on number of channels
   soundBuf = (float*) malloc(sizeof(float) * fileHeader->channels);
   if(frameStart <= 1){
      frameStart = 1;
   }
   //Set to frameStart, which is 1 indexed
   sf_seek(fileStream, frameStart-1, SEEK_SET);
   status = updateState(0,parent->getDeltaTime());
   //Reset filepointer to reread the same frame on the 0th timestep
   sf_seek(fileStream, frameStart-1, SEEK_SET);
   return status;
}

int SoundStream::updateState(double time, double dt){
   int status = PV_SUCCESS;
   assert(fileStream);
    
    // if (time >= nextSampleTime) {
    //     nextSampleTime += (1.0 / sampleRate);
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
       }
   // }
    
   return status;
}

PV::BaseObject * createSoundStream(char const * name, PV::HyPerCol * hc) {
   return hc ? new SoundStream(name, hc) : NULL;
} 

}  /* namespace PVsound */
