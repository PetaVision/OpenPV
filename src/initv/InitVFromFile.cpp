/*
 * InitVFromFile.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "InitVFromFile.hpp"
#include "components/LayerGeometry.hpp"
#include "io/FileManager.hpp"
#include "io/BroadcastLayerFile.hpp"
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
         ioFlag, getName(), "Vfilename", &mVfilename, nullptr, true /*warnIfAbsent*/);
   if (mVfilename == nullptr) {
      Fatal().printf(
            "InitVFromFile::initialize, group \"%s\": for InitVFromFile, string parameter "
            "\"Vfilename\" "
            "must be defined.  Exiting\n",
            getName());
   }
}

void InitVFromFile::ioParam_frameNumber(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, getName(), "frameNumber", &mFrameNumber, mFrameNumber, true /*warnIfAbsent*/);
}

Response::Status InitVFromFile::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = BaseInitV::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *objectTable = message->mObjectTable;
   FatalIf(
         objectTable == nullptr,
         "%s CommunicateInitInfo message sent with null object table\n",
         getDescription_c());
   auto *layerGeometry = objectTable->findObject<LayerGeometry>(getName());
   FatalIf(
         layerGeometry == nullptr,
         "%s could not find a LayerGeometry object\n",
         getDescription_c());
   if (!layerGeometry->getInitInfoCommunicatedFlag()) {
      InfoLog().printf(
            "%s must wait until the LayerGeometry component finishes its CommunicateInitInfo stage.\n",
            getDescription_c());
      return Response::POSTPONE;
   }
   mBroadcastFlag = layerGeometry->getBroadcastFlag();

   return Response::SUCCESS;
}

void InitVFromFile::calcV(float *V, const PVLayerLoc *loc) {
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
      if (mBroadcastFlag) {
         pvAssert(loc->nx == 1 and loc->ny == 1);
         BroadcastLayerFile inputLayerFile(
               fileManager,
               base,
               loc->nf,
               loc->nbatch,
               true /*readOnlyFlag*/,
               false /*clobberFlag*/,
               false /*verifyWritesFlag*/);
         for (int b = 0; b < loc->nbatch; ++b) {
            float *Vbatch = &V[b * loc->nx * loc->ny * loc->nf];
            inputLayerFile.setDataLocation(Vbatch, b);
         }
         inputLayerFile.setIndex(mFrameNumber);
         inputLayerFile.read();
      }
      else {
         LayerFile inputLayerFile(
               fileManager,
               base,
               *loc,
               false /*dataExtendedFlag*/,
               false /*fileExtendedFlag*/,
               true /*readOnlyFlag*/,
               false /*clobberFlag*/,
               false /*verifyWritesFlag*/);
         for (int b = 0; b < loc->nbatch; ++b) {
            float *Vbatch = &V[b * loc->nx * loc->ny * loc->nf];
            inputLayerFile.setDataLocation(Vbatch, b);
         }
         inputLayerFile.setIndex(mFrameNumber);
         inputLayerFile.read();
      }
   }
   else { // TODO: Treat as an image file
      if (fileManager->isRoot()) {
         ErrorLog().printf("InitVFromFile: file \"%s\" is not a pvp file.\n", this->mVfilename);
      }
      MPI_Barrier(fileManager->getMPIBlock()->getComm());
      exit(EXIT_FAILURE);
   }
}

} // end namespace PV
