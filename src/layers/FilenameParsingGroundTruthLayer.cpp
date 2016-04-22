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

#ifdef PV_USE_GDAL

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
   ioParam_gtClassTrueValue(ioFlag);
   ioParam_gtClassFalseValue(ioFlag);
   return status1;
}

void FilenameParsingGroundTruthLayer::ioParam_gtClassTrueValue(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "gtClassTrueValue", &gtClassTrueValue, 1.0f, false);
}


void FilenameParsingGroundTruthLayer::ioParam_gtClassFalseValue(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "gtClassFalseValue", &gtClassFalseValue, -1.0f, false);
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

int FilenameParsingGroundTruthLayer::communicateInitInfo() {
   movieLayer = dynamic_cast<Movie *>(parent->getLayerFromName(movieLayerName));
   if(movieLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: movieLayerName \"%s\" is not a layer in the HyPerCol.\n",
            getKeyword(), name, movieLayerName); 
      }
      exit(EXIT_FAILURE);
   }
   // What should the return value actually be? Unknown, but if we don't return a value
   // here, the program will abort with a hard to diagnose error.
   return PV_SUCCESS;
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

   for(int b = 0; b < loc->nbatch; b++){
      char * currentFilename = NULL;
      int filenameLen = 0;
      //TODO depending on speed of this layer, more efficient way would be to preallocate currentFilename buffer
      if(parent->icCommunicator()->commRank()==0){
         currentFilename = strdup(movieLayer->getFilename(b));
         //Get length of currentFilename and broadcast
         int filenameLen = (int) strlen(currentFilename) + 1; //+1 for the null terminator
         //Using local communicator, as each batch MPI will handle it's own run
         MPI_Bcast(&filenameLen, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
         //Braodcast filename to all other local processes
         MPI_Bcast(currentFilename, filenameLen, MPI_CHAR, 0, parent->icCommunicator()->communicator());
      }
      else{
         //Receive broadcast about length of filename
         MPI_Bcast(&filenameLen, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
         currentFilename = (char*)calloc(sizeof(char), filenameLen);
         //Receive filename
         MPI_Bcast(currentFilename, filenameLen, MPI_CHAR, 0, parent->icCommunicator()->communicator());
      }

      std::string fil = currentFilename;
      pvdata_t * ABatch = A + b * getNumExtended();
      for(int i = 0; i < num_neurons; i++){
         int nExt = kIndexExtended(i, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         int fi = featureIndex(nExt, loc->nx+loc->halo.rt+loc->halo.lt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
         int match = fil.find(classes[i]);
         if(0 <= match){
            ABatch[nExt] = gtClassTrueValue;
         }
         else{
            ABatch[nExt] = gtClassFalseValue;
         }
      }
      //Free buffer, TODO, preallocate buffer to avoid this
      free(currentFilename);
   }
   update_timer->stop();
   return PV_SUCCESS;
}

#else // PV_USE_GDAL
FilenameParsingGroundTruthLayer::FilenameParsingGroundTruthLayer(const char * name, HyPerCol * hc)
{
   if (hc->columnId()==0) {
      fprintf(stderr, "FilenameParsingGroundTruthLayer class requires compiling with PV_USE_GDAL set\n");
   }
   MPI_Barrier(hc->icCommunicator()->communicator());
   exit(EXIT_FAILURE);
}
FilenameParsingGroundTruthLayer::FilenameParsingGroundTruthLayer() {}
#endif // PV_USE_GDAL

BaseObject * createFilenameParsingGroundTruthLayer(char const * name, HyPerCol * hc) {
   return hc ? new FilenameParsingGroundTruthLayer(name, hc) : NULL;
}

} /* namespace PV */
