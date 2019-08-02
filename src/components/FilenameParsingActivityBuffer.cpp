/*
 * FilenameParsingActivityBuffer.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */

#include "FilenameParsingActivityBuffer.hpp"

#include "components/InputActivityBuffer.hpp"
#include "components/InputLayerNameParam.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace PV {
FilenameParsingActivityBuffer::FilenameParsingActivityBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

FilenameParsingActivityBuffer::~FilenameParsingActivityBuffer() {
   free(mInputLayerName);
   free(mClassListFileName);
}

int FilenameParsingActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_classList(ioFlag);
   ioParam_gtClassTrueValue(ioFlag);
   ioParam_gtClassFalseValue(ioFlag);
   return status;
}

void FilenameParsingActivityBuffer::ioParam_gtClassTrueValue(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "gtClassTrueValue", &mGtClassTrueValue, 1.0f, false);
}

void FilenameParsingActivityBuffer::ioParam_gtClassFalseValue(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "gtClassFalseValue", &mGtClassFalseValue, -1.0f, false);
}

void FilenameParsingActivityBuffer::ioParam_classList(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, name, "classList", &mClassListFileName, mClassListFileName, false);
   if (mClassListFileName == nullptr) {
      WarnLog() << getName()
                << ": No classList specified. Looking for classes.txt in output directory.\n";
   }
}

Response::Status FilenameParsingActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *objectTable = message->mObjectTable;
   if (mInputLayer == nullptr) {
      auto *inputLayerNameParam = objectTable->findObject<InputLayerNameParam>(getName());
      FatalIf(
            inputLayerNameParam == nullptr,
            "%s could not find an InputLayerNameParam component.\n",
            getDescription_c());
      if (!inputLayerNameParam->getInitInfoCommunicatedFlag()) {
         return Response::POSTPONE;
      }
      char const *linkedObjectName = inputLayerNameParam->getLinkedObjectName();
      mInputLayer                  = objectTable->findObject<InputLayer>(linkedObjectName);
      FatalIf(
            mInputLayer == nullptr,
            "%s inputLayerName \"%s\" points to an object that is not an InputLayer.\n",
            getDescription_c(),
            inputLayerNameParam->getLinkedObjectName());
   }
   if (!mInputLayer->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   auto *phaseParam = objectTable->findObject<PhaseParam>(getName());
   FatalIf(phaseParam == nullptr, "%s does not have a PhaseParam component.\n", getDescription_c());
   if (!phaseParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   int const phase = phaseParam->getPhase();

   auto *inputLayerPhaseParam = objectTable->findObject<PhaseParam>(mInputLayer->getName());
   int const inputLayerPhase  = inputLayerPhaseParam->getPhase();
   FatalIf(
         inputLayerPhase <= phase,
         "%s: The phase of layer %s (%d) must be greater than the phase of the "
         "FilenameParsingActivityBuffer (%d)\n",
         getDescription_c(),
         mInputLayerName,
         inputLayerPhase,
         phase);
   return Response::SUCCESS;
}

Response::Status FilenameParsingActivityBuffer::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   // Fill the mClasses vector.
   // This needs to go in registerData and not allocateDataStructures because the default path
   // for the class list is the output path, which is available only in registerData through
   // the checkpointer argument. But is that the best default?
   auto status = ActivityBuffer::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }

   std::ifstream inputFile;
   std::string outPath("");

   if (mClassListFileName != nullptr) {
      outPath += std::string(mClassListFileName);
   }
   else {
      auto *checkpointer = message->mDataRegistry;
      outPath += checkpointer->getOutputPath();
      outPath += "/classes.txt";
   }

   inputFile.open(outPath.c_str(), std::ifstream::in);
   FatalIf(!inputFile.is_open(), "%s: Unable to open file %s\n", getName(), outPath.c_str());

   mClasses.clear();
   std::string line;
   while (getline(inputFile, line)) {
      mClasses.push_back(line);
   }
   inputFile.close();

   std::size_t numFeatures = (std::size_t)getLayerLoc()->nf;
   FatalIf(
         numFeatures != mClasses.size(),
         "%s has %d features but classList \"%s\" has %zu categories.\n",
         getDescription_c(),
         getLayerLoc()->nf,
         outPath.c_str(),
         mClasses.size());
   return Response::SUCCESS;
}

void FilenameParsingActivityBuffer::updateBufferCPU(double time, double dt) {
   float *A                  = mBufferData.data();
   const PVLayerLoc *loc     = getLayerLoc();
   int numNeurons            = loc->nx * loc->ny * loc->nf;
   int const localBatchWidth = getLayerLoc()->nbatch;
   int const blockBatchWidth = getMPIBlock()->getBatchDimension() * localBatchWidth;
   for (int b = 0; b < blockBatchWidth; b++) {
      int const mpiBlockBatchIndex = b / localBatchWidth; // integer division
      int const localBatchIndex    = b % localBatchWidth;

      std::vector<float> fileMatches(mClasses.size());
      if (getMPIBlock()->getRank() == 0) {
         auto *inputActivityComponent = mInputLayer->getComponentByType<ActivityComponent>();
         pvAssert(inputActivityComponent);
         auto *inputActivityBuffer =
               inputActivityComponent->getComponentByType<InputActivityBuffer>();
         pvAssert(inputActivityBuffer);
         std::string currentFilename =
               inputActivityBuffer->getCurrentFilename(localBatchIndex, mpiBlockBatchIndex).c_str();
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

      float *ABatch = A + localBatchIndex * getBufferSize();
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
}

} // end namespace PV
