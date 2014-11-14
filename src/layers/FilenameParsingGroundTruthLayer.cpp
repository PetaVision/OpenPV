/*
 * FilenameParsingGroundTruthLayer.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */

#include "FilenameParsingGroundTruthLayer.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
namespace PV {

FilenameParsingGroundTruthLayer::FilenameParsingGroundTruthLayer(const char * name, HyPerCol * hc)
{
   initialize(name, hc);
}

FilenameParsingGroundTruthLayer::~FilenameParsingGroundTruthLayer()
{
   free(classes);
   delete classes;
   free(movieLayerName);
   free(movieLayer);
}

int FilenameParsingGroundTruthLayer::initialize(const char * name, HyPerCol * hc)
{
   int status = ANNLayer::initialize(name, hc);
   return status;
}

int FilenameParsingGroundTruthLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status1 = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_classes(ioFlag);
   ioParam_movieLayerName(ioFlag);
   movieLayer = dynamic_cast<Movie *>(parent->getLayerFromName(movieLayerName));
   if(movieLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: movieLayerName \"%s\" is not a layer in the HyPerCol.\n",
            parent->parameters()->groupKeywordFromName(name), name, movieLayerName); 
      }
   exit(EXIT_FAILURE);
   }
   return status1;
}

void FilenameParsingGroundTruthLayer::ioParam_movieLayerName(enum ParamsIOFlag ioFlag) {
      parent->ioParamStringRequired(ioFlag, name, "movieLayerName", &movieLayerName);

}

void FilenameParsingGroundTruthLayer::ioParam_classes(enum ParamsIOFlag ioFlag) {
   std::string outPath = parent->getOutputPath();
   outPath = outPath + "/classes.txt";
   inputfile.open(outPath.c_str(), std::ifstream::in);
   if (!inputfile.is_open()){
      std::cout << "Unable to open file " << outPath << "\n";
      exit(EXIT_FAILURE);
   }
   int i = 0;
   std::string line;
   while(getline(inputfile, line))
   {
      i++;
   }
   numClasses = i;
   inputfile.close();
   inputfile.open(outPath.c_str(), std::ifstream::in);   
   classes = new std::string[numClasses];
   for(i = 0 ; i < numClasses ; i++)
   {
      getline(inputfile, classes[i]);
   }
   inputfile.close();
}

bool FilenameParsingGroundTruthLayer::needUpdate(double time, double dt){
   bool movieUpdate =  movieLayer->needUpdate(parent->simulationTime(), parent->getDeltaTime());
   return movieUpdate;
}

int FilenameParsingGroundTruthLayer::updateState(double time, double dt)
{
   update_timer->start();
      pvdata_t * A = getCLayer()->activity->data;
      const PVLayerLoc * loc = getLayerLoc();
      int num_neurons = getNumNeurons();
      if (num_neurons != numClasses)
      {
         std::cout << "The number of neurons in " << getName() << " is not equal to the number of classes specified in " << parent->getOutputPath() << "/classes.txt\n";
         exit(EXIT_FAILURE);
      }   
      const char * currentFilename = movieLayer->getFilename();
      std::string fil = strdup(currentFilename);
      
      for(int i = 0; i < num_neurons; i++){
         int nExt = kIndexExtended(i, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         int fi = featureIndex(nExt, loc->nx+loc->halo.rt+loc->halo.lt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
         int match = fil.find(classes[i]);
         if(0 <= match){
            A[nExt] = 1;
         }
         else{
            A[nExt] = -1;
         }
      }
      update_timer->stop();
   return PV_SUCCESS;
}

} /* namespace PV */
