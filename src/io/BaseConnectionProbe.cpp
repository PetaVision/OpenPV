/*
 * BaseConnectionProbe.cpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#include "BaseConnectionProbe.hpp"

namespace PV {

BaseConnectionProbe::BaseConnectionProbe() {
   initialize_base();
}

BaseConnectionProbe::~BaseConnectionProbe() {
   free(name);
   assert(stream != NULL);
   if (stream->isfile) PV_fclose(stream);
   free(stream);
   free(targetConnName);
   free(probeOutputFile);
}

int BaseConnectionProbe::initialize_base() {
   name = NULL;
   stream = NULL;
   targetConnName = NULL;
   targetConn = NULL;
   probeOutputFile = NULL;
   return PV_SUCCESS;
}

int BaseConnectionProbe::initialize(const char * probename, HyPerCol * hc) {
   int status = PV_SUCCESS;
   if (hc==NULL) {
      fprintf(stderr, "BaseConnectionProbe error: probename cannot be null.\n");
      exit(EXIT_FAILURE);
   }
   parent = hc;

   if (probename==NULL) {
      fprintf(stderr, "BaseConnectionProbe error in rank %d process: probename cannot be null.\n", hc->columnId());
      status = PV_FAILURE;
   }
#if PV_USE_MPI
   MPI_Barrier(hc->icCommunicator()->communicator());
#endif
   if (status!=PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   name = strdup(probename);
   if (name==NULL) {
      fprintf(stderr, "BaseConnectionProbe \"%s\" error in rank %d process: unable to allocate memory for name of probe.\n", probename, hc->columnId());
      status = PV_FAILURE;
   }
#if PV_USE_MPI
   MPI_Barrier(hc->icCommunicator()->communicator());
#endif
   if (status!=PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   return status;
}

int BaseConnectionProbe::communicate() {
   int status = PV_SUCCESS;
   targetConn = getParent()->getConnFromName(getTargetConnName());
   if (targetConn==NULL) {
      fprintf(stderr, "BaseConnectionProbe \"%s\" error in rank %d process: targetConnection \"%s\" is not a connection in the HyPerCol.\n",
            name, getParent()->columnId(), getTargetConnName());
      status = PV_FAILURE;
   }
#if PV_USE_MPI
   MPI_Barrier(getParent()->icCommunicator()->communicator());
#endif
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   return status;
}

int BaseConnectionProbe::allocateProbe() {
   return PV_SUCCESS;
}


int BaseConnectionProbe::ioParams(enum ParamsIOFlag ioFlag) {
   parent->ioParamsStartGroup(ioFlag, name);
   ioParamsFillGroup(ioFlag);
   parent->ioParamsFinishGroup(ioFlag);
   return PV_SUCCESS;
}

int BaseConnectionProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_probeOutputFile(ioFlag);
   ioParam_targetConnection(ioFlag);
   return PV_SUCCESS;
}

void BaseConnectionProbe::ioParam_probeOutputFile(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "probeOutputFile", &probeOutputFile, NULL, true);
   if (ioFlag == PARAMS_IO_READ && parent->columnId()==0) {
      if( probeOutputFile != NULL && probeOutputFile[0] != '\0') {
         if (probeOutputFile[0] == '/') {
            stream = PV_fopen(probeOutputFile, "w");
         }
         else {
            char * outputdir = parent->getOutputPath();
            char * path = (char *) malloc(strlen(outputdir)+1+strlen(probeOutputFile)+1);
            sprintf(path, "%s/%s", outputdir, probeOutputFile);
            stream = PV_fopen(path, "w");
            if( !stream ) {
               fprintf(stderr, "BaseConnectionProbe error opening \"%s\" for writing: %s\n", path, strerror(errno));
               exit(EXIT_FAILURE);
            }
            free(path);
         }
      }
      else {
         stream = PV_stdout();
         if( !stream ) {
            exit(EXIT_FAILURE);
         }
      }
   }
}

void BaseConnectionProbe::ioParam_targetConnection(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "targetConnection", &targetConnName);
}

}  // end of namespace PV


