/*
 * DatastoreDelayTestLayer.cpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#include "DatastoreDelayTestLayer.hpp"
#include <include/pv_arch.h>

namespace PV {

DatastoreDelayTestLayer::DatastoreDelayTestLayer(const char* name, HyPerCol * hc) {
   initialize(name, hc);
}

DatastoreDelayTestLayer::~DatastoreDelayTestLayer() {}

int DatastoreDelayTestLayer::initialize(const char * name, HyPerCol * hc) {
   ANNLayer::initialize(name, hc);
   inited = false; // The first call to updateV sets this to true, so that the class knows whether to initialize or not.
   period = -1; // Can't set period until number of delay levels is determined, but that's determined by the connections,
                // which can't be created until after initialization is finished.  period will be set in the first call to updateV
   return PV_SUCCESS;
}

int DatastoreDelayTestLayer::updateState(double timed, double dt) {
   const PVLayerLoc * loc = getLayerLoc();
   return updateState(timed, dt, getNumNeurons(), getV(), getActivity(), loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
}

int DatastoreDelayTestLayer::updateState(double timef, double dt, int num_neurons, pvdata_t * V, pvdata_t * A, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   // updateV();
   updateV_DatastoreDelayTestLayer(getLayerLoc(), &inited, getV(), parent->icCommunicator()->publisherStore(getLayerId())->numberOfLevels());
   setActivity_HyPerLayer(parent->getNBatch(), num_neurons, A, V, nx, ny, nf, lt, rt, dn, up);
   // resetGSynBuffers(); // Since V doesn't use the GSyn buffers, no need to maintain them.

   return PV_SUCCESS;
}

int DatastoreDelayTestLayer::updateV_DatastoreDelayTestLayer(const PVLayerLoc * loc, bool * inited, pvdata_t * V, int period) {
   for(int b = 0; b < loc->nbatch; b++){
      pvdata_t * VBatch = V + b * loc->nx * loc->ny * loc->nf;
      if( *inited ) {
         // Rotate values by one row.
         // Move everything down one row; clobbering row 0 in the process
         for( int y=loc->ny-1; y>0; y-- ) {
            for( int x=0; x < loc->nx; x++ ) {
               for( int f=0; f < loc->nf; f++ ) {
                  pvdata_t * V1 = &VBatch[kIndex(x,y,f,loc->nx,loc->ny,loc->nf)];
                  (*V1)--;
                  if( *V1 == 0 ) *V1 = period;
               }
            }
         }
         // Finally, copy period-th row to zero-th row
         for( int x=0; x < loc->nx; x++ ) {
            for( int f=0; f < loc->nf; f++ ) {
               VBatch[kIndex(x,0,f,loc->nx,loc->ny,loc->nf)] = VBatch[kIndex(x,period,f,loc->nx,loc->ny,loc->nf)];
            }
         }

      }
      else {
         if( loc->ny < period ) {
#ifdef PV_USE_MPI
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
            int rank = 0;
#endif // PV_USE_MPI

            if( rank == 0 ) {
               fflush(stdout);
               fprintf(stderr, "DatastoreDelayTestLayer: number of rows (%d) must be >= period (%d).  Exiting.\n", loc->ny, period);
            }
            abort();
         }
         int base = loc->ky0;
         for( int x=0; x < loc->nx; x++ ) {
            for( int f=0; f < loc->nf; f++ ) {
               for( int row=0; row < loc->ny; row++ ) {
                  VBatch[kIndex(x,row,f,loc->nx,loc->ny,loc->nf)] = (base+row) % period + 1;
               }
            }
         }
         *inited = true;
      }
   }
   return PV_SUCCESS;
}

//int DatastoreDelayTestLayer::updateV() {
//   const PVLayerLoc * loc = getLayerLoc();
//   int nx = loc->nx;
//   int ny = loc->ny;
//   int nf = loc->nf;
//   if( inited ) {
//      // Rotate values by one row.
//      // Move everything down one row; clobbering row 0 in the process
//      for( int y=ny-1; y>0; y-- ) {
//         for( int x=0; x<nx; x++ ) {
//            for( int f=0; f<nf; f++ ) {
//               pvdata_t * V = &getV()[kIndex(x,y,f,nx,ny,nf)];
//               (*V)--;
//               if( *V == 0 ) *V = period;
//            }
//         }
//      }
//      // Finally, copy period-th row to zero-th row
//      for( int x=0; x<loc->nx; x++ ) {
//         for( int f=0; f<loc->nf; f++ ) {
//            getV()[kIndex(x,0,f,nx,ny,nf)] = getV()[kIndex(x,period,f,nx,ny,nf)];
//         }
//      }
//   }
//   else {
//      period = parent->icCommunicator()->publisherStore(clayer->layerId)->numberOfLevels();
//      if( nx < period ) {
//         if( parent->icCommunicator()->commRank() == 0 ) {
//            fflush(stdout);
//            fprintf(stderr, "DatastoreDelayTestLayer \"%s\": number of columns (%d) must be >= number of delay levels (%d).  Exiting.\n", name, getLayerLoc()->ny, period);
//         }
//         exit(EXIT_FAILURE);
//      }
//      int base = loc->ky0;
//      for( int x=0; x<nx; x++ ) {
//         for( int f=0; f<nf; f++ ) {
//            for( int row=0; row<ny; row++ ) {
//               getV()[kIndex(x,row,f,nx,ny,nf)] = (base+row) % period + 1;
//            }
//         }
//      }
//      inited = true;
//   }
//   return PV_SUCCESS;
// }

}  // end of namespace PV block

