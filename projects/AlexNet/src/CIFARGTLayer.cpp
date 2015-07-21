/*
 * CIFARGTLayer.cpp
 * Author: slundquist
 */

#include "CIFARGTLayer.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

namespace PV {
CIFARGTLayer::CIFARGTLayer(const char * name, HyPerCol * hc)
{
   negativeGt = false;
   //constantValue = false;
   //firstRun = true;
   iVal = -1;

   initialize(name, hc);
}

CIFARGTLayer::~CIFARGTLayer(){
   //if(inputfile) inputfile.close();
}

int CIFARGTLayer::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();

   Communicator * comm = parent->icCommunicator();

   HyPerLayer* baseLayer = parent->getLayerFromName(imageLayerName);
   if (baseLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: imageLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, imageLayerName);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }

   imageLayer = dynamic_cast<Image*>(baseLayer);
   if (imageLayer ==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: imageLayerName \"%s\" is not a subclass of Image .\n",
                 parent->parameters()->groupKeywordFromName(name), name, imageLayerName);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   return status;
}


int CIFARGTLayer::initialize(const char * name, HyPerCol * hc) {
   //TODO make only root process do this
   //Is there a way to implement a test for mpi?
   int status = ANNLayer::initialize(name, hc);
   
   ////2 files are test and train, assuming name of the layer is either test or train
   ////std::string filename = "input/" + std::string(name) + ".txt";
   //inputfile.open(inFilename, std::ifstream::in);
   //if (!inputfile.is_open()){
   //   std::cout << "Unable to open file " << inFilename << "\n";
   //   exit(EXIT_FAILURE);
   //}
   //if(startFrame < 1){
   //   std::cout << "Setting startFrame to 1\n";
   //   startFrame = 1;
   //}
   ////Skip for startFrame
   //for(int i = 0; i < startFrame; i++){
   //   getline (inputfile,inputString);
   //}
   return status;
}

int CIFARGTLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   //ioParam_inFilename(ioFlag);
   //ioParam_StartFrame(ioFlag);
   ioParam_NegativeGt(ioFlag);
   ioParam_ImageLayerName(ioFlag);
   //ioParam_constantValue(ioFlag);
   return status;
}

//void CIFARGTLayer::ioParam_inFilename(enum ParamsIOFlag ioFlag) {
//   parent->ioParamStringRequired(ioFlag, name, "inFilename", &inFilename);
//}
//
//void CIFARGTLayer::ioParam_StartFrame(enum ParamsIOFlag ioFlag) {
//   parent->ioParamValue(ioFlag, name, "startFrame", &startFrame, startFrame);
//}

void CIFARGTLayer::ioParam_ImageLayerName(enum ParamsIOFlag ioFlag){
   parent->ioParamStringRequired(ioFlag, name, "imageLayerName", &imageLayerName);
}


void CIFARGTLayer::ioParam_NegativeGt(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "negativeGt", &negativeGt, negativeGt);
}

//void CIFARGTLayer::ioParam_constantValue(enum ParamsIOFlag ioFlag) {
//   parent->ioParamValue(ioFlag, name, "constantValue", &constantValue, constantValue);
//}

int CIFARGTLayer::updateState(double timef, double dt) {
   //getline (inputfile,inputString);
   inputString = std::string(imageLayer->getFilename());
   unsigned found = inputString.find_last_of("/\\");
   //CIFAR is 0 indexed
   char cVal = inputString.at(found-1);
   iVal = cVal - '0';

   pvdata_t * A = getCLayer()->activity->data;
   const PVLayerLoc * loc = getLayerLoc(); 

   //std::cout << "time: " << parent->simulationTime() << " inputString:" << inputString << "  iVal:" << iVal << "\n";
   assert(iVal >= 0 && iVal < 10);
   //NF must be 10, one for each class
   assert(loc->nf == 10);
   for(int ni = 0; ni < getNumNeurons(); ni++){
      int nExt = kIndexExtended(ni, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      int fi = featureIndex(nExt, loc->nx+loc->halo.rt+loc->halo.lt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
      if(fi == iVal){
         A[nExt] = 1;
      }
      else{
         if(negativeGt){
            A[nExt] = -1;
         }
         else{
            A[nExt] = 0;
         }
      }
   }
   return PV_SUCCESS;
}
}
