/*
 * InitVFromFile.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "InitVFromFile.hpp"
#include "columns/HyPerCol.hpp"
#include "utils/BufferUtilsMPI.hpp"

namespace PV {
InitVFromFile::InitVFromFile() { initialize_base(); }

InitVFromFile::InitVFromFile(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitVFromFile::~InitVFromFile() { free(mVfilename); }

int InitVFromFile::initialize_base() { return PV_SUCCESS; }

int InitVFromFile::initialize(char const *name, HyPerCol *hc) {
   int status = BaseInitV::initialize(name, hc);
   return status;
}

int InitVFromFile::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInitV::ioParamsFillGroup(ioFlag);
   ioParam_Vfilename(ioFlag);
   return status;
}

void InitVFromFile::ioParam_Vfilename(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "Vfilename", &mVfilename, nullptr, true /*warnIfAbsent*/);
   if (mVfilename == nullptr) {
      Fatal().printf(
            "InitVFromFile::initialize, group \"%s\": for InitVFromFile, string parameter "
            "\"Vfilename\" "
            "must be defined.  Exiting\n",
            name);
   }
}

int InitVFromFile::registerData(Checkpointer *checkpointer, std::string const &objName) {
   /* nothing to checkpoint, but we need the MPIBlock to scatter the data in calcV. */
   mMPIBlock = checkpointer->getMPIBlock();
   return PV_SUCCESS;
}

int InitVFromFile::calcV(float *V, const PVLayerLoc *loc) {
   char const *ext = strrchr(mVfilename, '.');
   bool isPvpFile  = (ext && strcmp(ext, ".pvp") == 0);
   if (isPvpFile) {
      FileStream fileStream(mVfilename, std::ios_base::in | std::ios_base::binary, false);
      BufferUtils::ActivityHeader header = BufferUtils::readActivityHeader(fileStream);
      int fileType                       = header.fileType;
      if (header.fileType == PVP_NONSPIKING_ACT_FILE_TYPE) {
         readDenseActivityPvp(V, loc, fileStream, header);
      }
      else { // TODO: Handle sparse activity pvp files.
         if (mMPIBlock->getRank() == 0) {
            ErrorLog() << "InitVFromFile: filename \"" << mVfilename << "\" has fileType "
                       << header.fileType << ", which is not supported for InitVFromFile.\n";
         }
         MPI_Barrier(mMPIBlock->getComm());
         MPI_Finalize();
         exit(EXIT_FAILURE);
      }
   }
   else { // TODO: Treat as an image file
      if (mMPIBlock->getRank() == 0) {
         ErrorLog().printf("InitVFromFile: file \"%s\" is not a pvp file.\n", this->mVfilename);
      }
      MPI_Barrier(mMPIBlock->getComm());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

void InitVFromFile::readDenseActivityPvp(
      float *V,
      PVLayerLoc const *loc,
      FileStream &fileStream,
      BufferUtils::ActivityHeader const &header) {
   bool isRootProc         = mMPIBlock->getRank() == 0;
   std::size_t frameSize   = (std::size_t)header.recordSize * sizeof(float) + sizeof(double);
   int numFrames           = header.nBands;
   int blockBatchDimension = mMPIBlock->getBatchDimension();
   for (int m = 0; m < blockBatchDimension; m++) {
      for (int b = 0; b < loc->nbatch; b++) {
         int globalBatchIndex = (mMPIBlock->getStartBatch() + m) * loc->nbatch + b;
         float *Vbatch        = V + b * (loc->nx * loc->ny * loc->nf);
         Buffer<float> pvpBuffer;
         if (isRootProc) {
            int frameNumber = globalBatchIndex % numFrames;
            fileStream.setOutPos(sizeof(header) + frameNumber * sizeof(float) * frameSize, true);
            int xStart = header.nx * mMPIBlock->getStartColumn() / mMPIBlock->getNumColumns();
            int yStart = header.ny * mMPIBlock->getStartRow() / mMPIBlock->getNumRows();
            pvpBuffer.resize(header.nx, header.ny, header.nf);
            BufferUtils::readFrameWindow(fileStream, &pvpBuffer, header, xStart, yStart, 0);
         }
         else {
            pvpBuffer.resize(loc->nx, loc->ny, loc->nf);
         }
         BufferUtils::scatter(mMPIBlock, pvpBuffer, loc->nx, loc->ny, m, 0);
         std::vector<float> bufferData = pvpBuffer.asVector();
         std::memcpy(Vbatch, bufferData.data(), sizeof(float) * pvpBuffer.getTotalElements());
      }
   }
}

} // end namespace PV
