/*
 * InitV.cpp
 *
 *  Created on: Dec 6, 2011
 *      Author: pschultz
 */

#include "InitV.hpp"
//#include "Image.hpp"

namespace PV {
InitV::InitV() {
   initialize_base();
   // Default constructor for derived classes.  A derived class should call
   // InitV::initialize from its own initialize routine, A derived class's
   // constructor should not call a non-default InitV constructor.
}

InitV::InitV(HyPerCol *hc, const char *groupName) {
   initialize_base();
   initialize(hc, groupName);
}

int InitV::initialize_base() {
   groupName       = NULL;
   initVTypeString = NULL;
   initVTypeCode   = UndefinedInitV;
   filename        = NULL;
   return PV_SUCCESS;
}

int InitV::initialize(HyPerCol *hc, const char *groupName) {
   this->parent    = hc;
   this->groupName = strdup(groupName);
   return PV_SUCCESS;
}

InitV::~InitV() {
   free(this->groupName);
   free(this->initVTypeString);
   free(this->filename);
}

void InitV::ioParamGroup_ConstantV(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, groupName, "valueV", &constantValue, (pvdata_t)V_REST);
}

void InitV::ioParamGroup_ZeroV(enum ParamsIOFlag ioFlag) { constantValue = 0.0f; }

void InitV::ioParamGroup_UniformRandomV(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, groupName, "minV", &minV, 0.0f);
   parent->parameters()->ioParamValue(ioFlag, groupName, "maxV", &maxV, minV + 1.0f);
}

void InitV::ioParamGroup_GaussianRandomV(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, groupName, "meanV", &meanV, 0.0f);
   parent->parameters()->ioParamValue(ioFlag, groupName, "sigmaV", &sigmaV, 1.0f);
}

void InitV::ioParamGroup_InitVFromFile(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, groupName, "Vfilename", &filename, NULL, true /*warnIfAbsent*/);
   if (filename == NULL) {
      initVTypeCode = UndefinedInitV;
      pvErrorNoExit().printf(
            "InitV::initialize, group \"%s\": for InitVFromFile, string parameter \"Vfilename\" "
            "must be defined.  Exiting\n",
            groupName);
      abort();
   }
}

int InitV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status  = PV_SUCCESS;
   printErrors = parent->getCommunicator()->commRank() == 0;
   parent->parameters()->ioParamString(
         ioFlag, groupName, "InitVType", &initVTypeString, "ConstantV", true /*warnIfAbsent*/);
   if (!strcmp(initVTypeString, "ConstantV")) {
      initVTypeCode = ConstantV;
      ioParamGroup_ConstantV(ioFlag);
   }
   else if (!strcmp(initVTypeString, "ZeroV")) {
      initVTypeCode = ConstantV;
      ioParamGroup_ZeroV(ioFlag);
   }
   else if (!strcmp(initVTypeString, "UniformRandomV")) {
      initVTypeCode = UniformRandomV;
      ioParamGroup_UniformRandomV(ioFlag);
   }
   else if (!strcmp(initVTypeString, "GaussianRandomV")) {
      initVTypeCode = GaussianRandomV;
      ioParamGroup_GaussianRandomV(ioFlag);
   }
   else if (!strcmp(initVTypeString, "InitVFromFile")) {
      initVTypeCode = InitVFromFile;
      ioParamGroup_InitVFromFile(ioFlag);
   }
   else {
      initVTypeCode = UndefinedInitV;
      if (printErrors)
         pvErrorNoExit().printf(
               "InitV::initialize, group \"%s\": InitVType \"%s\" not recognized.\n",
               groupName,
               initVTypeString);
      abort();
   }
   return status;
}

int InitV::calcV(HyPerLayer *layer) {
   int status            = PV_SUCCESS;
   const PVLayerLoc *loc = layer->getLayerLoc();
   pvdata_t *V           = layer->getV();
   if (V == NULL) {
      pvErrorNoExit().printf(
            "%s: InitV called but membrane potential V is null.\n", layer->getDescription_c());
      exit(EXIT_FAILURE);
   }
   switch (initVTypeCode) {
      case UndefinedInitV:
         status = PV_FAILURE;
         if (printErrors)
            pvErrorNoExit().printf("InitV::calcV: InitVType was undefined.\n");
         break;
      case ConstantV: status       = calcConstantV(V, layer->getNumNeuronsAllBatches()); break;
      case UniformRandomV: status  = calcUniformRandomV(V, loc, layer->getParent()); break;
      case GaussianRandomV: status = calcGaussianRandomV(V, loc, layer->getParent()); break;
      case InitVFromFile:
         status = calcVFromFile(V, layer->getLayerLoc(), layer->getParent()->getCommunicator());
         break;
      default:
         status = PV_FAILURE;
         if (printErrors)
            pvErrorNoExit().printf("InitV::calcV: InitVType was an unrecognized type.\n");
         break;
   }
   return status;
}

int InitV::calcConstantV(pvdata_t *V, int numNeurons) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int k = 0; k < numNeurons; k++)
      V[k]    = constantValue;
   return PV_SUCCESS;
}

int InitV::calcGaussianRandomV(pvdata_t *V, const PVLayerLoc *loc, HyPerCol *hc) {
   PVLayerLoc flatLoc;
   memcpy(&flatLoc, loc, sizeof(PVLayerLoc));
   flatLoc.nf                = 1;
   GaussianRandom *randState = new GaussianRandom(&flatLoc, false /*isExtended*/);
   const int nxny            = flatLoc.nx * flatLoc.ny;
   for (int b = 0; b < loc->nbatch; b++) {
      pvdata_t *VBatch = V + b * loc->nx * loc->ny * loc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int xy = 0; xy < nxny; xy++) {
         for (int f = 0; f < loc->nf; f++) {
            int index     = kIndex(xy, 0, f, nxny, 1, loc->nf);
            VBatch[index] = randState->gaussianDist(xy, meanV, sigmaV);
         }
      }
   }
   delete randState;
   return PV_SUCCESS;
}

int InitV::calcUniformRandomV(pvdata_t *V, const PVLayerLoc *loc, HyPerCol *hc) {
   PVLayerLoc flatLoc;
   memcpy(&flatLoc, loc, sizeof(PVLayerLoc));
   flatLoc.nf        = 1;
   Random *randState = new Random(&flatLoc, false /*isExtended*/);
   const int nxny    = flatLoc.nx * flatLoc.ny;
   for (int b = 0; b < loc->nbatch; b++) {
      pvdata_t *VBatch = V + b * loc->nx * loc->ny * loc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int xy = 0; xy < nxny; xy++) {
         for (int f = 0; f < loc->nf; f++) {
            int index     = kIndex(xy, 0, f, nxny, 1, loc->nf);
            VBatch[index] = randState->uniformRandom(xy, minV, maxV);
         }
      }
   }
   delete randState;
   return PV_SUCCESS;
}

int InitV::calcVFromFile(pvdata_t *V, const PVLayerLoc *loc, Communicator *icComm) {
   int status = PV_SUCCESS;
   PVLayerLoc fileLoc;
   char const *ext = strrchr(filename, '.');
   bool isPvpFile  = (ext && strcmp(ext, ".pvp") == 0);
   if (isPvpFile) {
      PV_Stream *readFile = pvp_open_read_file(filename, icComm);
      if (icComm->commRank() == 0) {
         if (readFile == NULL) {
            pvError().printf(
                  "InitV::calcVFromFile error: path \"%s\" could not be opened: %s.  Exiting.\n",
                  filename,
                  strerror(errno));
         }
      }
      else {
         assert(readFile == NULL); // Only root process should be doing input/output
      }
      assert(icComm->commRank() == 0 || readFile == NULL);
      assert(
            (readFile != NULL && icComm->commRank() == 0)
            || (readFile == NULL && icComm->commRank() != 0));
      int numParams = NUM_BIN_PARAMS;
      int params[NUM_BIN_PARAMS];
      int status = pvp_read_header(readFile, icComm, params, &numParams);
      if (status != PV_SUCCESS) {
         read_header_err(filename, icComm, numParams, params);
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
         if (icComm->commRank() == 0) {
            pvErrorNoExit().printf(
                  "InitVFromFilename: dimensions of \"%s\" (x=%d,y=%d,f=%d) do not agree with "
                  "layer dimensions (x=%d,y=%d,f=%d).\n",
                  filename,
                  params[INDEX_NX_GLOBAL],
                  params[INDEX_NY_GLOBAL],
                  params[INDEX_NF],
                  loc->nxGlobal,
                  loc->nyGlobal,
                  loc->nf);
         }
         MPI_Barrier(icComm->communicator());
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
         if (icComm->commRank() == 0) {
            pvErrorNoExit().printf(
                  "HyPerLayer::readBufferFile: file \"%s\" appears to be in an obsolete version of "
                  "the .pvp format.\n",
                  filename);
         }
         abort();
      }

      for (int b = 0; b < loc->nbatch; b++) {
         pvdata_t *VBatch = V + b * (loc->nx * loc->ny * loc->nf);
         switch (filetype) {
            case PVP_FILE_TYPE:
               pvErrorIf(
                     printErrors,
                     "calcVFromFile for file \"%s\": \"PVP_FILE_TYPE\" files is obsolete.\n",
                     this->filename);
               break;
            case PVP_ACT_FILE_TYPE:
               if (printErrors)
                  pvErrorNoExit().printf(
                        "calcVFromFile for file \"%s\": sparse activity files are not yet "
                        "implemented for initializing V buffers.\n",
                        this->filename);
               abort();
               break;
            case PVP_NONSPIKING_ACT_FILE_TYPE:
               double dummytime;
               pvp_read_time(readFile, icComm, 0 /*root process*/, &dummytime);
               status = scatterActivity(
                     readFile,
                     icComm,
                     0 /*root process*/,
                     VBatch,
                     loc,
                     false /*extended*/,
                     &fileLoc);
               break;
            default:
               if (printErrors)
                  pvErrorNoExit().printf(
                        "calcVFromFile: file \"%s\" is not an activity pvp file.\n",
                        this->filename);
               abort();
               break;
         }
      }
      pvp_close_file(readFile, icComm);
      readFile = NULL;
   }
   else { // Treat as an image file
      if (printErrors)
         pvErrorNoExit().printf("calcVFromFile: file \"%s\" is not a pvp file.\n", this->filename);
      abort();
   }
   return status;
}

int InitV::checkLoc(const PVLayerLoc *loc, int nx, int ny, int nf, int nxGlobal, int nyGlobal) {
   int status = PV_SUCCESS;
   if (checkLocValue(loc->nxGlobal, nxGlobal, "nxGlobal") != PV_SUCCESS)
      status = PV_FAILURE;
   if (checkLocValue(loc->nyGlobal, nyGlobal, "nyGlobal") != PV_SUCCESS)
      status = PV_FAILURE;
   if (checkLocValue(loc->nf, nf, "nf") != PV_SUCCESS)
      status = PV_FAILURE;
   return status;
}

int InitV::checkLocValue(int fromParams, int fromFile, const char *field) {
   int status = PV_SUCCESS;
   if (fromParams != fromFile) {
      if (printErrors)
         pvErrorNoExit().printf(
               "InitVFromFile: Incompatible %s: parameter group \"%s\" gives %d; filename \"%s\" "
               "gives %d\n",
               field,
               groupName,
               fromParams,
               filename,
               fromFile);
      status = PV_FAILURE;
   }
   return status;
}

} // end namespace PV
