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
}

ColProbe::ColProbe(const char * filename, HyPerCol * hc) { // Kept for backward compatibility
   initialize("ColProbe", filename, hc);
}

ColProbe::ColProbe(const char * probeName, const char * filename, HyPerCol * hc) {
    initialize(probeName, filename, hc);
}

ColProbe::~ColProbe() {
   if( fp != NULL && fp != stdout) {
      fclose(fp);
   }
}

int ColProbe::initialize(const char * probeName, const char * filename, HyPerCol * hc) {
   initialize_path(filename, hc);
   setColProbeName(probeName);
   return PV_SUCCESS;
}

int ColProbe::initialize_path(const char * filename, HyPerCol * hc) {
   if( filename != NULL ) {
      char * path;
      const char * output_path = hc->getOutputPath();
      size_t len = strlen(output_path) + strlen(filename) + 1;
      path = (char *) malloc( len * sizeof(char) );
      sprintf(path, "%s/%s", output_path, filename);
      fp = fopen(path, "w");
      free(path);
   }
   else {
      fp = stdout;
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
