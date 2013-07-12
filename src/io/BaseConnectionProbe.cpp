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
}

int BaseConnectionProbe::initialize_base() {
   name = NULL;
   stream = NULL;
   targetConnName = NULL;
   targetConn = NULL;
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
   MPI_Barrier(hc->icCommunicator()->communicator());
   if (status!=PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   name = strdup(probename);
   if (name==NULL) {
      fprintf(stderr, "BaseConnectionProbe \"%s\" error in rank %d process: unable to allocate memory for name of probe.\n", probename, hc->columnId());
      status = PV_FAILURE;
   }
   MPI_Barrier(hc->icCommunicator()->communicator());
   if (status!=PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   if (hc->columnId()==0) {
      const char * filename = hc->parameters()->stringValue(name, "probeOutputFile");
      if( filename != NULL ) {
         char * outputdir = hc->getOutputPath();
         char * path = (char *) malloc(strlen(outputdir)+1+strlen(filename)+1);
         sprintf(path, "%s/%s", outputdir, filename);
         stream = PV_fopen(path, "w");
         if( !stream ) {
            fprintf(stderr, "BaseConnectionProbe error opening \"%s\" for writing: %s\n", path, strerror(errno));
            exit(EXIT_FAILURE);
         }
         free(path);
      }
      else {
         stream = PV_stdout();
         if( !stream ) {
            exit(EXIT_FAILURE);
         }
      }
   }

   const char * target_conn_name = hc->parameters()->stringValue(name, "targetConnection");
   if (target_conn_name==NULL) {
      fprintf(stderr, "BaseConnectionProbe \"%s\" error in rank %d process: targetConnection cannot be null.\n", probename, hc->columnId());
      status = PV_FAILURE;
   }
   MPI_Barrier(hc->icCommunicator()->communicator());
   if (status!=PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   targetConnName = strdup(target_conn_name);
   if (target_conn_name==NULL) {
      fprintf(stderr, "BaseConnectionProbe \"%s\" error in rank %d process: unable to allocate memory for name of target connection.\n", probename, hc->columnId());
      status = PV_FAILURE;
   }
   MPI_Barrier(hc->icCommunicator()->communicator());
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
   MPI_Barrier(getParent()->icCommunicator()->communicator());
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   return status;
}

int BaseConnectionProbe::allocateProbe() {
   return PV_SUCCESS;
}

}  // end of namespace PV


