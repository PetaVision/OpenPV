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
      // This is not MPI-ically correct!
      // Move everything down one row; clobbering row 0 in the process
      for( int y=ny-1; y>0; y-- ) {
         for( int x=0; x<nx; x++ ) {
            for( int f=0; f<nf; f++ ) {
               getV()[kIndex(x,y,f,nx,ny,nf)] = getV()[kIndex(x,y-1,f,nx,ny,nf)];
            }
         }
      }
      // Finally, copy period-th row to zero-th row
      for( int x=0; x<loc->ny; x++ ) {
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
      for( int x=0; x<nx; x++ ) {
         for( int f=0; f<nf; f++ ) {
            for( int baserow=0; baserow<period; baserow++ ) {
               pvdata_t r = baserow+1;
               getV()[kIndex(x,baserow,f,nx,ny,nf)] = r;
            }
            for( int row=period; row<ny; row++ ) {
               int baserow=row % period;
               getV()[kIndex(x,row,f,nx,ny,nf)] = getV()[kIndex(x,baserow,f,nx,ny,nf)];
            }
         }
      }
      inited = true;
   }
   return PV_SUCCESS;
}

}  // end of namespace PV block

