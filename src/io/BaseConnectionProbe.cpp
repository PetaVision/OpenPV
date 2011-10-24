/*
 * BaseConnectionProbe.cpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#include "BaseConnectionProbe.hpp"
#include "ConnectionProbe.hpp"  // Temporary to allow backwards-compatibility kludge

namespace PV {

BaseConnectionProbe::BaseConnectionProbe() {
   initialize_base();
}

BaseConnectionProbe::~BaseConnectionProbe() {
   free(name);
   free(filename);
   assert(fp != NULL);
   if( fp != stdout && fp != NULL ) fclose(fp);
}

int BaseConnectionProbe::initialize_base() {
   name = NULL;
   filename = NULL;
   fp = NULL;
   return PV_SUCCESS;
}

int BaseConnectionProbe::initialize(const char * probename, const char * filename, HyPerCol * hc) {
   if( probename ) {
      name = strdup(probename);
   }
   else {
      name = strdup("Unnamed connection probe");
   }
   if( filename ) {
      this->filename = strdup(filename);
   }
   else {
      this->filename = NULL;
   }
                                       /* dynamic_cast below is temporary for backwards compatibility of ConnectionProbe */
   ConnectionProbe * connectionProbe = dynamic_cast<ConnectionProbe *>(this);
   if( ( hc && hc->icCommunicator()->commRank() == 0 ) || connectionProbe != NULL ) {
      if( filename ) {
         const char * outputdir = hc->getOutputPath();
         if( strlen(outputdir) + strlen(filename) + 2 > PV_PATH_MAX ) {
            fprintf(stderr, "BaseConnectionProbe: output filename \"%s/%s\" too long.  Exiting.\n",outputdir,filename);
            exit(EXIT_FAILURE);
         }
         char path[PV_PATH_MAX];
         sprintf(path, "%s/%s", outputdir, filename);

         fp = fopen(path, "w");
         if( fp == NULL )  {
            fprintf(stderr, "BaseConnectionProbe: unable to open \"%s\" for writing.  Error code %d.\n", path, errno);
            exit(EXIT_FAILURE);
         }
      }
      else {
         fp = stdout;
      }
   }
   else {
      fp = NULL;
   }
   return PV_SUCCESS;
}

}  // end of namespace PV


