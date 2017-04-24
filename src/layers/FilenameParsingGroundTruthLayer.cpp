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
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
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
   parent->parameters()->ioParamString(
         ioFlag, name, "classList", &mClassListFileName, mClassListFileName, false);
   if (mClassListFileName == nullptr) {
      WarnLog() << getName()
                << ": No classList specified. Looking for classes.txt in output directory.\n";
   }
}

int FilenameParsingGroundTruthLayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();

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

   std::size_t numFeatures = (std::size_t)getLayerLoc()->nf;
   FatalIf(
         numFeatures != mClasses.size(),
         "%s has %d features but classList \"%s\" has %zu categories.\n",
         getDescription_c(),
         getLayerLoc()->nf,
         outPath.c_str(),
         mClasses.size());
   return status;
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
   float *A                  = getCLayer()->activity->data;
   const PVLayerLoc *loc     = getLayerLoc();
   int numNeurons            = getNumNeurons();
   int const localBatchWidth = getLayerLoc()->nbatch;
   int const blockBatchWidth = getMPIBlock()->getBatchDimension() * localBatchWidth;
   for (int b = 0; b < blockBatchWidth; b++) {
      int const mpiBlockBatchIndex = b / localBatchWidth; // integer division
      int const localBatchIndex    = b % localBatchWidth;

      std::vector<float> fileMatches(mClasses.size());
      if (getMPIBlock()->getRank() == 0) {
         std::string currentFilename =
               mInputLayer->getCurrentFilename(localBatchIndex, mpiBlockBatchIndex).c_str();
         for (auto ci = (std::size_t)0; ci < mClasses.size(); ci++) {
            std::size_t match = currentFilename.find(mClasses.at(ci));
            fileMatches[ci]   = match != std::string::npos ? mGtClassTrueValue : mGtClassFalseValue;
         }
      }
      // It seems clunky to send each process all the fileMatches, when
      // they'll only use only the fileMatches for the correct MPIBlock
      // batch index.  Use MPI_Send/MPI_Recv?  Create more MPI_Comm's?
      MPI_Bcast(fileMatches.data(), fileMatches.size(), MPI_FLOAT, 0, getMPIBlock()->getComm());

      if (getMPIBlock()->getBatchIndex() != mpiBlockBatchIndex) {
         continue;
      }

      float *ABatch = A + localBatchIndex * getNumExtended();
      for (int i = 0; i < numNeurons; i++) {
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
         ABatch[nExt] = fileMatches[fi];
      }
   }
   update_timer->stop();
   return PV_SUCCESS;
}

} // end namespace PV
