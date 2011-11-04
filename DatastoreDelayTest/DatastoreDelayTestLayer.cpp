/*
 * DatastoreDelayTestLayer.cpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#include "DatastoreDelayTestLayer.hpp"

namespace PV {

DatastoreDelayTestLayer::DatastoreDelayTestLayer(const char* name, HyPerCol * hc) : ANNLayer(name, hc){
   initialize();
}

DatastoreDelayTestLayer::~DatastoreDelayTestLayer() {}

int DatastoreDelayTestLayer::initialize() {
   inited = false; // The first call to updateV sets this to true, so that the class knows whether to initialize or not.
   period = -1; // Can't set period until number of delay levels is determined, but that's determined by the connections,
                // which can't be created until after initialization is finished.  period will be set in the first call to updateV
   return PV_SUCCESS;
}

int DatastoreDelayTestLayer::updateV() {
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   if( inited ) {
      // Rotate values by one row.
      // Move everything down one row; clobbering row 0 in the process
      for( int y=ny-1; y>0; y-- ) {
         for( int x=0; x<nx; x++ ) {
            for( int f=0; f<nf; f++ ) {
               pvdata_t * V = &getV()[kIndex(x,y,f,nx,ny,nf)];
               (*V)--;
               if( *V == 0 ) *V = period;
            }
         }
      }
      // Finally, copy period-th row to zero-th row
      for( int x=0; x<loc->nx; x++ ) {
         for( int f=0; f<loc->nf; f++ ) {
            getV()[kIndex(x,0,f,nx,ny,nf)] = getV()[kIndex(x,period,f,nx,ny,nf)];
         }
      }
   }
   else {
      period = parent->icCommunicator()->publisherStore(clayer->layerId)->numberOfLevels();
      if( nx < period ) {
         if( parent->icCommunicator()->commRank() == 0 ) {
            fflush(stdout);
            fprintf(stderr, "DatastoreDelayTestLayer \"%s\": number of columns (%d) must be >= number of delay levels (%d).  Exiting.\n", name, getLayerLoc()->ny, period);
         }
         exit(EXIT_FAILURE);
      }
      int base = loc->ky0;
      for( int x=0; x<nx; x++ ) {
         for( int f=0; f<nf; f++ ) {
            for( int row=0; row<ny; row++ ) {
               getV()[kIndex(x,row,f,nx,ny,nf)] = (base+row) % period + 1;
            }
         }
      }
      inited = true;
   }
   return PV_SUCCESS;
}

}  // end of namespace PV block

