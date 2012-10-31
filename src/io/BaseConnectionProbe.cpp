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
   free(filename);
   assert(fp != NULL);
   if( fp != stdout && fp != NULL ) fclose(fp);
}

int BaseConnectionProbe::initialize_base() {
   name = NULL;
   filename = NULL;
   fp = NULL;
   targetConn = NULL;
   isPostProbe = 1; //default to 1
   return PV_SUCCESS;
}

int BaseConnectionProbe::initialize(const char * probename, const char * filename, HyPerConn * conn) {
   int kx = 0;
   int ky = 0;
   int kf = 0;
   bool postProbeFlag = true;

   initialize(probename,filename,conn,kx,ky,kf,postProbeFlag);

   return PV_SUCCESS;
}
int BaseConnectionProbe::initialize(const char * probename, const char * filename, HyPerConn * conn, int k, bool postProbeFlag) {

   int kx, ky, kf;
   const PVLayerLoc * loc;
   if (isPostProbe) {
      loc = conn->postSynapticLayer()->getLayerLoc();
   }
   else
   {
      loc = conn->preSynapticLayer()->getLayerLoc();
   }
   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   int nf = loc->nf;
   kx = kxPos(k,nxGlobal,nyGlobal,nf);
   ky = kyPos(k,nxGlobal,nyGlobal,nf);
   kf = featureIndex(k,nxGlobal,nyGlobal,nf);

   initialize(probename,filename,conn,kx,ky,kf,postProbeFlag);

   return PV_SUCCESS;
}

int BaseConnectionProbe::initialize(const char * probename, const char * filename, HyPerConn * conn, int kx, int ky, int kf, bool postProbeFlag) {
   isPostProbe = postProbeFlag;

   bool inBounds = false;
   const PVLayerLoc * loc;
   if (isPostProbe) {
      loc = conn->postSynapticLayer()->getLayerLoc();
   }
   else
   {
      loc = conn->preSynapticLayer()->getLayerLoc();
   }
   int kxLocal = kx - loc->kx0;
   int kyLocal = ky - loc->ky0;

   //Restricted index only for both post and pre
   inBounds = !(kxLocal < 0 || kxLocal >= loc->nx || kyLocal < 0 || kyLocal >= loc->ny);

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

   HyPerCol * hc = conn->getParent();
   if(inBounds ) { // if inBounds
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
            fprintf(stderr, "BaseConnectionProbe error opening \"%s\" for writing: %s.\n", path, strerror(errno));
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
   targetConn = conn;
   conn->insertProbe(this);
   return PV_SUCCESS;
}

}  // end of namespace PV


