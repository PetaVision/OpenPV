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
#include "io/SparseBroadcastLayerFile.hpp"
#include "io/SparseLayerFile.hpp"
#include "utils/PathComponents.hpp"

namespace PV {
InitVFromFile::InitVFromFile() { initialize_base(); }

InitVFromFile::InitVFromFile(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize_base();
   initialize(paramsIO, comm);
}

InitVFromFile::~InitVFromFile() {}

int InitVFromFile::initialize_base() { return PV_SUCCESS; }

void InitVFromFile::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   BaseInitV::initialize(paramsIO, comm);
}

int InitVFromFile::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = BaseInitV::ioParamsFillGroup(ioSwitch);
   ioParam_Vfilename(ioSwitch);
   ioParam_frameNumber(ioSwitch);
   return status;
}

void InitVFromFile::ioParam_Vfilename(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "Vfilename", &mVfilename);
   FatalIf(
         mVfilename.empty(),
         "InitVFromFile, group \"%s\": string parameter \"Vfilename\" must be defined. Exiting.\n",
         getName());
}

void InitVFromFile::ioParam_frameNumber(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "frameNumber", &mFrameNumber);
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
      int fileType;
      if (fileManager->isRoot()) {
         BufferUtils::ActivityHeader header = BufferUtils::readActivityHeader(*inputFile);
         fileType = header.fileType;
      }
      MPI_Bcast(
            &fileType,
            1 /*count*/,
            MPI_INT,
            fileManager->getRootProcessRank(),
            fileManager->getMPIBlock()->getComm());
      switch (fileType) {
         case PVP_ACT_SPARSEVALUES_FILE_TYPE:
            if (mBroadcastFlag) {
               readSparseBroadcastLayerFile(V, fileManager, base, *loc);
            }
            else {
               readSparseLayerFile(V, fileManager, base, *loc);
            }
            break;
         case PVP_NONSPIKING_ACT_FILE_TYPE:
            if (mBroadcastFlag) {
               readBroadcastLayerFile(V, fileManager, base, *loc);
            }
            else {
               readLayerFile(V, fileManager, base, *loc);
            }
            break;
         default:
            Fatal().printf(
                  "InitVFromFile \"%s\" is not an activity file (file type %d)\n",
                  mVfilename.c_str(), fileType);
            break;
      }
   }
   else { // TODO: Treat as an image file
      if (fileManager->isRoot()) {
         ErrorLog().printf("InitVFromFile: file \"%s\" is not a pvp file.\n", mVfilename.c_str());
      }
      MPI_Barrier(fileManager->getMPIBlock()->getComm());
      std::exit(EXIT_FAILURE);
   }
}

void InitVFromFile::readBroadcastLayerFile(
      float *V,
      std::shared_ptr<FileManager> fileManager,
      std::string const &filename,
      PVLayerLoc const &loc) {
   pvAssert(loc.nx == 1 and loc.ny == 1);
   BroadcastLayerFile inputLayerFile(
         fileManager,
         filename,
         loc.nf,
         loc.nbatch,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWritesFlag*/);
   for (int b = 0; b < loc.nbatch; ++b) {
      float *Vbatch = &V[b * loc.nx * loc.ny * loc.nf];
      inputLayerFile.setDataLocation(Vbatch, b);
   }
   inputLayerFile.setIndex(mFrameNumber);
   inputLayerFile.read();
}

void InitVFromFile::readLayerFile(
      float *V,
      std::shared_ptr<FileManager> fileManager,
      std::string const &filename,
      PVLayerLoc const &loc) {
   LayerFile inputLayerFile(
         fileManager,
         filename,
         loc,
         false /*dataExtendedFlag*/,
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWritesFlag*/);
   for (int b = 0; b < loc.nbatch; ++b) {
      float *Vbatch = &V[b * loc.nx * loc.ny * loc.nf];
      inputLayerFile.setDataLocation(Vbatch, b);
   }
   inputLayerFile.setIndex(mFrameNumber);
   inputLayerFile.read();
}
void InitVFromFile::readSparseBroadcastLayerFile(
      float *V,
      std::shared_ptr<FileManager> fileManager,
      std::string const &filename,
      PVLayerLoc const &loc) {
   pvAssert(loc.nx == 1 and loc.ny == 1);
   SparseBroadcastLayerFile inputLayerFile(
         fileManager,
         filename,
         loc.nf,
         loc.nbatch,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<SparseList<float>> sparseLists(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      inputLayerFile.setListLocation(&sparseLists.at(b), b);
   }
   inputLayerFile.setIndex(mFrameNumber);
   inputLayerFile.read();
   for (int b = 0; b < loc.nbatch; ++b) {
      float *Vbatch = &V[b * loc.nf];
      std::vector<SparseList<float>::Entry> sparseContents = sparseLists[b].getContents();
      for (auto const &entry : sparseContents) {
         int index = entry.index;
         FatalIf(
               index >= loc.nf or index < 0,
               "SparseBroadcastLayerFile \"%s\" batch element %d has index %d, "
               "which is out of bounds for a 1-by-1-by-%d layer.\n",
               filename.c_str(), b, index, loc.nf);
         float value = entry.value;
         Vbatch[index] = value;
      }
   }
}

void InitVFromFile::readSparseLayerFile(
      float *V,
      std::shared_ptr<FileManager> fileManager,
      std::string const &filename,
      PVLayerLoc const &loc) {
   SparseLayerFile inputLayerFile(
         fileManager,
         filename,
         loc,
         false /*dataExtendedFlag*/,
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWritesFlag*/);
   std::vector<SparseList<float>> sparseLists(loc.nbatch);
   for (int b = 0; b < loc.nbatch; ++b) {
      inputLayerFile.setListLocation(&sparseLists.at(b), b);
   }
   inputLayerFile.setIndex(mFrameNumber);
   inputLayerFile.read();
   for (int b = 0; b < loc.nbatch; ++b) {
      int neuronsPerBatchElement = loc.nx * loc.ny * loc.nf;
      float *Vbatch = &V[b * neuronsPerBatchElement];
      std::vector<SparseList<float>::Entry> contents = sparseLists[b].getContents();
      for (auto const &entry : contents) {
         int index = entry.index;
         FatalIf(
               index >= neuronsPerBatchElement or index < 0,
               "SparseLayerFile \"%s\" batch element %d has index %d, which is out of bounds "
               "for a %d-by-%d-by-%d layer.\n",
               filename.c_str(), b, index, loc.nx, loc.ny, loc.nf);
         float value = entry.value;
         Vbatch[index] = value;
      }
   }
}

} // end namespace PV
