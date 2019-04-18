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
   ioParam_frameNumber(ioFlag);
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

void InitVFromFile::ioParam_frameNumber(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "frameNumber", &mFrameNumber, mFrameNumber, true /*warnIfAbsent*/);
}

void InitVFromFile::calcV(float *V, const PVLayerLoc *loc) {
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
         if (getMPIBlock()->getRank() == 0) {
            ErrorLog() << "InitVFromFile: filename \"" << mVfilename << "\" has fileType "
                       << header.fileType << ", which is not supported for InitVFromFile.\n";
         }
         MPI_Barrier(getMPIBlock()->getComm());
         MPI_Finalize();
         exit(EXIT_FAILURE);
      }
   }
   else { // TODO: Treat as an image file
      if (getMPIBlock()->getRank() == 0) {
         ErrorLog().printf("InitVFromFile: file \"%s\" is not a pvp file.\n", this->mVfilename);
      }
      MPI_Barrier(getMPIBlock()->getComm());
      exit(EXIT_FAILURE);
   }
}

void InitVFromFile::readDenseActivityPvp(
      float *V,
      PVLayerLoc const *loc,
      FileStream &fileStream,
      BufferUtils::ActivityHeader const &header) {
   auto mpiBlock           = getMPIBlock();
   bool isRootProc         = mpiBlock->getRank() == 0;
   std::size_t frameSize   = (std::size_t)header.recordSize * sizeof(float) + sizeof(double);
   int numFrames           = header.nBands;
   int blockBatchDimension = mpiBlock->getBatchDimension();
   for (int m = 0; m < blockBatchDimension; m++) {
      for (int b = 0; b < loc->nbatch; b++) {
         int globalBatchIndex = (mpiBlock->getStartBatch() + m) * loc->nbatch + b;
         float *Vbatch        = V + b * (loc->nx * loc->ny * loc->nf);
         Buffer<float> pvpBuffer;
         if (isRootProc) {
            int frameIndex = (mFrameNumber + globalBatchIndex) % numFrames;
            fileStream.setOutPos(sizeof(header) + frameIndex * sizeof(float) * frameSize, true);
            int xStart = header.nx * mpiBlock->getStartColumn() / mpiBlock->getNumColumns();
            int yStart = header.ny * mpiBlock->getStartRow() / mpiBlock->getNumRows();
            pvpBuffer.resize(header.nx, header.ny, header.nf);
            BufferUtils::readFrameWindow(fileStream, &pvpBuffer, header, xStart, yStart, 0);
         }
         else {
            pvpBuffer.resize(loc->nx, loc->ny, loc->nf);
         }
         BufferUtils::scatter(mpiBlock, pvpBuffer, loc->nx, loc->ny, m, 0);
         std::vector<float> bufferData = pvpBuffer.asVector();
         std::memcpy(Vbatch, bufferData.data(), sizeof(float) * pvpBuffer.getTotalElements());
      }
   }
}

} // end namespace PV
