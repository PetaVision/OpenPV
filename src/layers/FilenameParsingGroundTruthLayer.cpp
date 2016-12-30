/*
 * FilenameParsingGroundTruthLayer.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */

#include "FilenameParsingGroundTruthLayer.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace PV {
FilenameParsingGroundTruthLayer::FilenameParsingGroundTruthLayer(const char *name, HyPerCol *hc) {
   initialize(name, hc);
}

FilenameParsingGroundTruthLayer::~FilenameParsingGroundTruthLayer() {
   free(mInputLayerName);
   free(mClassListFileName);
}

int FilenameParsingGroundTruthLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_classList(ioFlag);
   ioParam_inputLayerName(ioFlag);
   ioParam_gtClassTrueValue(ioFlag);
   ioParam_gtClassFalseValue(ioFlag);
   return status;
}

void FilenameParsingGroundTruthLayer::ioParam_gtClassTrueValue(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "gtClassTrueValue", &mGtClassTrueValue, 1.0f, false);
}

void FilenameParsingGroundTruthLayer::ioParam_gtClassFalseValue(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "gtClassFalseValue", &mGtClassFalseValue, -1.0f, false);
}

void FilenameParsingGroundTruthLayer::ioParam_inputLayerName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "inputLayerName", &mInputLayerName);
}

void FilenameParsingGroundTruthLayer::ioParam_classList(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(ioFlag, name, "classList", &mClassListFileName, mClassListFileName, false);
   if (mClassListFileName == nullptr) {
      WarnLog() << getName() << ": No classList specified. Looking for classes.txt in output directory.\n";
   }
}

int FilenameParsingGroundTruthLayer::allocateDataStructures() {
   ANNLayer::allocateDataStructures();

   std::ifstream mInputFile;
   std::string outPath("");

   if (mClassListFileName != nullptr) {   
      outPath += std::string(mClassListFileName);
   }
   else {
      outPath += parent->getOutputPath();
      outPath += "/classes.txt";
   }

   mInputFile.open(outPath.c_str(), std::ifstream::in);
   FatalIf(!mInputFile.is_open(), "%s: Unable to open file %s\n", getName(), outPath.c_str());

   mClasses.clear();
   std::string line;
   while (getline(mInputFile, line)) {
      mClasses.push_back(line);
   }
   mInputFile.close();
}

int FilenameParsingGroundTruthLayer::communicateInitInfo() {
   mInputLayer = dynamic_cast<InputLayer *>(parent->getLayerFromName(mInputLayerName));
   FatalIf(
         mInputLayer == nullptr && parent->columnId() == 0,
         "%s: inputLayerName \"%s\" is not a layer in the HyPerCol.\n",
         getDescription_c(),
         mInputLayerName);
   FatalIf(
         mInputLayer->getPhase() <= getPhase(),
         "%s: The phase of layer %s (%d) must be greater than the phase of the "
         "FilenameParsingGroundTruthLayer (%d)\n",
         getName(),
         mInputLayerName,
         mInputLayer->getPhase(),
         getPhase());
   return PV_SUCCESS;
}

bool FilenameParsingGroundTruthLayer::needUpdate(double time, double dt) {
   return mInputLayer->needUpdate(parent->simulationTime(), parent->getDeltaTime());
}

int FilenameParsingGroundTruthLayer::updateState(double time, double dt) {
   update_timer->start();
   float *A              = getCLayer()->activity->data;
   const PVLayerLoc *loc = getLayerLoc();
   int num_neurons       = getNumNeurons();
   FatalIf(
         num_neurons != mClasses.size(),
         "The number of neurons in %s is not equal to the number of classes specified in "
         "%s/classes.txt\n",
         getName(),
         parent->getOutputPath());

   for (int b = 0; b < loc->nbatch; ++b) {
      char *currentFilename = nullptr;
      int filenameLen       = 0;
      if (parent->getCommunicator()->commRank() == 0) {
         currentFilename = strdup(mInputLayer->getFileName(b).c_str());
         int filenameLen = (int)strlen(currentFilename) + 1; // +1 for the null terminator
         MPI_Bcast(&filenameLen, 1, MPI_INT, 0, parent->getCommunicator()->communicator());
         // Braodcast filename to all other local processes
         MPI_Bcast(
               currentFilename,
               filenameLen,
               MPI_CHAR,
               0,
               parent->getCommunicator()->communicator());
      }
      else {
         // Receive broadcast about length of filename
         MPI_Bcast(&filenameLen, 1, MPI_INT, 0, parent->getCommunicator()->communicator());
         currentFilename = (char *)calloc(sizeof(char), filenameLen);
         // Receive filename
         MPI_Bcast(
               currentFilename,
               filenameLen,
               MPI_CHAR,
               0,
               parent->getCommunicator()->communicator());
      }

      std::string fil = currentFilename;
      float *ABatch   = A + b * getNumExtended();
      for (int i = 0; i < num_neurons; i++) {
         int nExt = kIndexExtended(
               i,
               loc->nx,
               loc->ny,
               loc->nf,
               loc->halo.lt,
               loc->halo.rt,
               loc->halo.dn,
               loc->halo.up);
         int fi = featureIndex(
               nExt,
               loc->nx + loc->halo.rt + loc->halo.lt,
               loc->ny + loc->halo.dn + loc->halo.up,
               loc->nf);
         int match = fil.find(mClasses.at(i));
         if (0 <= match) {
            ABatch[nExt] = mGtClassTrueValue;
         }
         else {
            ABatch[nExt] = mGtClassFalseValue;
         }
      }
      free(currentFilename);
   }
   update_timer->stop();
   return PV_SUCCESS;
}
}
