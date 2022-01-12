/*
 * InitVFromFile.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "InitVFromFile.hpp"
#include "io/FileManager.hpp"
#include "io/LayerFile.hpp"
#include "utils/PathComponents.hpp"

namespace PV {
InitVFromFile::InitVFromFile() { initialize_base(); }

InitVFromFile::InitVFromFile(char const *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

InitVFromFile::~InitVFromFile() { free(mVfilename); }

int InitVFromFile::initialize_base() { return PV_SUCCESS; }

void InitVFromFile::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseInitV::initialize(name, params, comm);
}

int InitVFromFile::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInitV::ioParamsFillGroup(ioFlag);
   ioParam_Vfilename(ioFlag);
   ioParam_frameNumber(ioFlag);
   return status;
}

void InitVFromFile::ioParam_Vfilename(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
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
   parameters()->ioParamValue(
         ioFlag, name, "frameNumber", &mFrameNumber, mFrameNumber, true /*warnIfAbsent*/);
}

void InitVFromFile::calcV(float *V, const PVLayerLoc *loc) {
   auto ioMPIBlock  = getCommunicator()->getIOMPIBlock();
   std::string dir  = dirName(mVfilename);
   std::string base = baseName(mVfilename);
   std::string ext  = extension(mVfilename);
   auto fileManager = std::make_shared<FileManager>(getCommunicator()->getGlobalMPIBlock(), dir);

   bool isPvpFile   = (ext == ".pvp");
   if (isPvpFile) {
      auto inputFile = fileManager->open(base, std::ios_base::in, false);
      if (fileManager->isRoot()) {
         BufferUtils::ActivityHeader header = BufferUtils::readActivityHeader(*inputFile);
         int fileType                       = header.fileType;
         FatalIf(
               fileType != PVP_NONSPIKING_ACT_FILE_TYPE,
               "filename \"%s\" has fileType %d,  which is not supported for InitVFromFile.\n",
               mVfilename, fileType);
      }
      LayerFile inputLayerFile(fileManager, base, *loc, false, false, true, false);
      // booleans are dataExtended=false, fileExtended=false, readOnly=true, verifyWrites=false
      for (int b = 0; b < loc->nbatch; ++b) {
         float *Vbatch = &V[b * loc->nx * loc->ny * loc->nf];
         inputLayerFile.setDataLocation(Vbatch, b);
      }
      inputLayerFile.setIndex(mFrameNumber);
      inputLayerFile.read();
   }
   else { // TODO: Treat as an image file
      if (ioMPIBlock->getRank() == 0) {
         ErrorLog().printf("InitVFromFile: file \"%s\" is not a pvp file.\n", this->mVfilename);
      }
      MPI_Barrier(ioMPIBlock->getComm());
      exit(EXIT_FAILURE);
   }
}

} // end namespace PV
