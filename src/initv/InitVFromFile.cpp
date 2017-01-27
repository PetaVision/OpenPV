/*
 * InitVFromFile.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "InitVFromFile.hpp"
#include "columns/HyPerCol.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"

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

int InitVFromFile::calcV(float *V, const PVLayerLoc *loc) {
   int status = PV_SUCCESS;
   PVLayerLoc fileLoc;
   Communicator *comm = parent->getCommunicator();
   bool isRootProc = comm->commRank() == 0;
   char const *ext = strrchr(mVfilename, '.');
   bool isPvpFile  = (ext && strcmp(ext, ".pvp") == 0);
   if (isPvpFile) {
      FileStream headerStream(mVfilename, std::ios_base::in | std::ios_base::binary, false);
      struct BufferUtils::ActivityHeader header = BufferUtils::readActivityHeader(headerStream);
      int fileType = header.fileType;
      for (int b = 0; b < loc->nbatch; b++) {
         float *Vbatch = V + b * (loc->nx * loc->ny * loc->nf);
         Buffer<float> pvpBuffer;
         if (isRootProc) {
            BufferUtils::readDenseFromPvp(mVfilename, &pvpBuffer, b);
         }
         else {
            pvpBuffer.resize(loc->nx, loc->ny, loc->nf);
         }
         BufferUtils::scatter(comm, pvpBuffer, loc->nx, loc->ny);
         std::vector<float> bufferData = pvpBuffer.asVector();
         std::memcpy(Vbatch, bufferData.data(), sizeof(float) * pvpBuffer.getTotalElements());
      }
   }
   else { // TODO: Treat as an image file
      if (isRootProc) {
         ErrorLog().printf("InitVFromFile: file \"%s\" is not a pvp file.\n", this->mVfilename);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

int InitVFromFile::checkLoc(
      const PVLayerLoc *loc,
      int nx,
      int ny,
      int nf,
      int nxGlobal,
      int nyGlobal) {
   int status = PV_SUCCESS;
   if (checkLocValue(loc->nxGlobal, nxGlobal, "nxGlobal") != PV_SUCCESS)
      status = PV_FAILURE;
   if (checkLocValue(loc->nyGlobal, nyGlobal, "nyGlobal") != PV_SUCCESS)
      status = PV_FAILURE;
   if (checkLocValue(loc->nf, nf, "nf") != PV_SUCCESS)
      status = PV_FAILURE;
   return status;
}

int InitVFromFile::checkLocValue(int fromParams, int fromFile, const char *field) {
   int status = PV_SUCCESS;
   if (fromParams != fromFile) {
      if (parent->getCommunicator()->commRank() == 0) {
         ErrorLog().printf(
               "InitVFromFile: Incompatible %s: parameter group \"%s\" gives %d; "
               "filename \"%s\" gives %d\n",
               field,
               name,
               fromParams,
               mVfilename,
               fromFile);
      }
      status = PV_FAILURE;
   }
   return status;
}

} // end namespace PV
