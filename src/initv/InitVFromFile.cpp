/*
 * InitVFromFile.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "InitVFromFile.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {
InitVFromFile::InitVFromFile() {
   initialize_base();
}

InitVFromFile::InitVFromFile(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

InitVFromFile::~InitVFromFile() {
   free(mVfilename);
}

int InitVFromFile::initialize_base() {
   return PV_SUCCESS;
}

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
      pvError().printf(
            "InitVFromFile::initialize, group \"%s\": for InitVFromFile, string parameter \"Vfilename\" "
            "must be defined.  Exiting\n",
            name);
   }
}

int InitVFromFile::calcV(pvdata_t *V, const PVLayerLoc *loc) {
   int status = PV_SUCCESS;
   PVLayerLoc fileLoc;
   bool isRootProc = parent->getCommunicator()->commRank()==0;
   char const *ext = strrchr(mVfilename, '.');
   bool isPvpFile  = (ext && strcmp(ext, ".pvp") == 0);
   if (isPvpFile) {
      PV_Stream *readFile = pvp_open_read_file(mVfilename, parent->getCommunicator());
      if (parent->getCommunicator()->commRank() == 0) {
         if (readFile == NULL) {
            pvError().printf(
                  "InitVFromFile::calcVFromFile error: path \"%s\" could not be opened: %s.  Exiting.\n",
                  mVfilename,
                  strerror(errno));
         }
      }
      else {
         assert(readFile == NULL); // Only root process should be doing input/output
      }
      assert(parent->getCommunicator()->commRank() == 0 || readFile == NULL);
      assert(
            (readFile != NULL && parent->getCommunicator()->commRank() == 0)
            || (readFile == NULL && parent->getCommunicator()->commRank() != 0));
      int numParams = NUM_BIN_PARAMS;
      int params[NUM_BIN_PARAMS];
      int status = pvp_read_header(readFile, parent->getCommunicator(), params, &numParams);
      if (status != PV_SUCCESS) {
         read_header_err(mVfilename, parent->getCommunicator(), numParams, params);
      }
      int filetype = params[INDEX_FILE_TYPE];
      status       = checkLoc(
            loc,
            params[INDEX_NX],
            params[INDEX_NY],
            params[INDEX_NF],
            params[INDEX_NX_GLOBAL],
            params[INDEX_NY_GLOBAL]);
      if (status != PV_SUCCESS) {
         if (parent->getCommunicator()->commRank() == 0) {
            pvErrorNoExit().printf(
                  "InitVFromFilename: dimensions of \"%s\" (x=%d,y=%d,f=%d) do not agree with "
                  "layer dimensions (x=%d,y=%d,f=%d).\n",
                  mVfilename,
                  params[INDEX_NX_GLOBAL],
                  params[INDEX_NY_GLOBAL],
                  params[INDEX_NF],
                  loc->nxGlobal,
                  loc->nyGlobal,
                  loc->nf);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      fileLoc.nx       = params[INDEX_NX];
      fileLoc.ny       = params[INDEX_NY];
      fileLoc.nf       = params[INDEX_NF];
      fileLoc.nxGlobal = params[INDEX_NX_GLOBAL];
      fileLoc.nyGlobal = params[INDEX_NY_GLOBAL];
      fileLoc.kx0      = 0;
      fileLoc.ky0      = 0;
      if (params[INDEX_NX_PROCS] != 1 || params[INDEX_NY_PROCS] != 1) {
         if (parent->getCommunicator()->commRank() == 0) {
            pvErrorNoExit().printf(
                  "HyPerLayer::readBufferFile: file \"%s\" appears to be in an obsolete version of "
                  "the .pvp format.\n",
                  mVfilename);
         }
         abort();
      }

      for (int b = 0; b < loc->nbatch; b++) {
         pvdata_t *VBatch = V + b * (loc->nx * loc->ny * loc->nf);
         switch (filetype) {
            case PVP_FILE_TYPE:
               if (isRootProc) {
                  pvError().printf(
                        "calcVFromFile for file \"%s\": \"PVP_FILE_TYPE\" files is obsolete.\n",
                        this->mVfilename);
               }
               break;
            case PVP_ACT_FILE_TYPE:
               if (isRootProc)
                  pvErrorNoExit().printf(
                        "calcVFromFile for file \"%s\": sparse activity files are not yet "
                        "implemented for initializing V buffers.\n",
                        this->mVfilename);
               exit(EXIT_FAILURE);
               break;
            case PVP_NONSPIKING_ACT_FILE_TYPE:
               double dummytime;
               pvp_read_time(readFile, parent->getCommunicator(), 0 /*root process*/, &dummytime);
               status = scatterActivity(
                     readFile,
                     parent->getCommunicator(),
                     0 /*root process*/,
                     VBatch,
                     loc,
                     false /*extended*/,
                     &fileLoc);
               break;
            default:
               if (isRootProc)
                  pvErrorNoExit().printf(
                        "calcVFromFile: file \"%s\" is not an activity pvp file.\n",
                        this->mVfilename);
               abort();
               break;
         }
      }
      pvp_close_file(readFile, parent->getCommunicator());
      readFile = NULL;
   }
   else { // Treat as an image file
      if (isRootProc)
         pvErrorNoExit().printf("calcVFromFile: file \"%s\" is not a pvp file.\n", this->mVfilename);
      abort();
   }
   return status;
}

int InitVFromFile::checkLoc(const PVLayerLoc *loc, int nx, int ny, int nf, int nxGlobal, int nyGlobal) {
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
      if (parent->getCommunicator()->commRank()==0) {
         pvErrorNoExit().printf(
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
