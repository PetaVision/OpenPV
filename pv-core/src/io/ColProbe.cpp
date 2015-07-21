/*
 * ColProbe.cpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#include "ColProbe.hpp"

namespace PV {

ColProbe::ColProbe() { // Default constructor to be called by derived classes.
   // They should call ColProbe::initialize from their own initialization routine
   // instead of calling a non-default constructor.
   initialize_base();
}

ColProbe::ColProbe(const char * probeName, HyPerCol * hc) {
   initialize_base();
   initialize(probeName, hc);
}

ColProbe::~ColProbe() {
   if (stream != NULL && stream->isfile) {
      PV_fclose(stream);
   }
   free(colProbeName);
}

int ColProbe::initialize_base() {
   parentCol = NULL;
   stream = NULL;
   colProbeName = NULL;
   return PV_SUCCESS;
}

int ColProbe::initialize(const char * probeName, HyPerCol * hc) {
   parentCol = hc;
   int status = setColProbeName(probeName);
   if (status==PV_SUCCESS) {
      status = ioParamsFillGroup(PARAMS_IO_READ);
   }
   if (status==PV_SUCCESS) {
      status = hc->insertProbe(this);
   }
   return status;
}

int ColProbe::ioParams(enum ParamsIOFlag ioFlag) {
   parentCol->ioParamsStartGroup(ioFlag, colProbeName);
   ioParamsFillGroup(ioFlag);
   parentCol->ioParamsFinishGroup(ioFlag);
   return PV_SUCCESS;
}

int ColProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_probeOutputFile(ioFlag);
   return PV_SUCCESS;
}

void ColProbe::ioParam_probeOutputFile(enum ParamsIOFlag ioFlag) {
   char * filename = NULL;
   parentCol->ioParamString(ioFlag, colProbeName, "probeOutputFile", &filename, NULL, false/*warnIfAbsent*/);
   if (ioFlag==PARAMS_IO_READ) {
      initialize_stream(filename);
   }
   free(filename); filename = NULL;
}

int ColProbe::initialize_stream(const char * filename) {
   if (parentCol->columnId()==0)
   {
      if( filename != NULL ) {
         char * path;
         const char * output_path = parentCol->getOutputPath();
         size_t len = strlen(output_path) + strlen(filename) + 2; // One char for slash; one for string terminator
         path = (char *) malloc( len * sizeof(char) );
         sprintf(path, "%s/%s", output_path, filename);
         stream = PV_fopen(path, "w", parentCol->getVerifyWrites());
         free(path);
      }
      else {
         stream = PV_stdout();
      }
   }
   else {
      stream = NULL;
   }
   return PV_SUCCESS;
}

int ColProbe::setColProbeName(const char * name) {
   colProbeName = (char *) malloc(strlen(name) + 1);
   if( colProbeName ) {
      strcpy(colProbeName, name);
      return PV_SUCCESS;
   }
   else {
      fprintf(stderr, "Unable to allocate memory for ColProbe name \"%s\"\n", name);
      return PV_FAILURE;
   }
}

}  // end namespace PV
