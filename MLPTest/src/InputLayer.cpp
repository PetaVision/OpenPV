/*
 * InputLayer.cpp
 * Author: slundquist
 */

#include "InputLayer.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

namespace PV {
InputLayer::InputLayer(const char * name, HyPerCol * hc)
{
   initialize(name, hc);
}


int InputLayer::initialize(const char * name, HyPerCol * hc) {
   //TODO make only root process do this
   //Is there a way to implement a test for mpi?
   int status = ANNLayer::initialize(name, hc);
   //2 files are test and train, assuming name of the layer is either test or train
   //std::string filename = "input/" + std::string(name) + ".txt";
   std::ifstream inputfile (inFilename);
   if (inputfile.is_open())
   {
      getline (inputfile,inputString);
      inputfile.close();
   }
   else{
      std::cout << "Unable to open file " << inFilename << "\n";
      exit(EXIT_FAILURE);
   }
   assert(parent->getStartTime() == 0);
   assert(parent->getDeltaTime() == 1);
   numExamples = inputString.length();
   return status;
}

int InputLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_inFilename(ioFlag);
   return status;
}

void InputLayer::ioParam_inFilename(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "inFilename", &inFilename);
}

int InputLayer::updateState(double timef, double dt) {
   char cVal = inputString.at(int(parent->simulationTime()-1)%numExamples);
   int iVal = cVal - '0';
   pvdata_t * A = getCLayer()->activity->data;
   const PVLayerLoc * loc = getLayerLoc(); 
   assert(loc->nf == 2);
   //Set binary values of xor values
   std::cout << timef << ": input val:" << iVal << "\n";
   for(int ni = 0; ni < getNumNeurons(); ni++){
      int nExt = kIndexExtended(ni, loc->nx, loc->ny, loc->nf, loc->nb);
      int fi = featureIndex(nExt, loc->nx+2*loc->nb, loc->ny+2*loc->nb, loc->nf);
      switch(iVal){
         case 0:
            if(fi == 0){
               A[nExt] = 0;
            }
            if(fi == 1){
               A[nExt] = 0;
            }
            break;
         case 1:
            if(fi == 0){
               A[nExt] = 0;
            }
            if(fi == 1){
               A[nExt] = 1;
            }
            break;
         case 2:
            if(fi == 0){
               A[nExt] = 1;
            }
            if(fi == 1){
               A[nExt] = 0;
            }
            break;
         case 3:
            if(fi == 0){
               A[nExt] = 1;
            }
            if(fi == 1){
               A[nExt] = 1;
            }
            break;
      }
   }
   return PV_SUCCESS;
}

}
