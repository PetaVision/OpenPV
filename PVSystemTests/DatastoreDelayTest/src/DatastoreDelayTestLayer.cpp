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
   if( *inited ) {
      for(int b = 0; b < loc->nbatch; b++){
         pvdata_t * VBatch = V + b * loc->nx * loc->ny * loc->nf;
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
   }
   else {
      for(int b = 0; b < loc->nbatch; b++){
         pvdata_t * VBatch = V + b * loc->nx * loc->ny * loc->nf;
         if( loc->ny < period ) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

BaseObject * createDatastoreDelayTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new DatastoreDelayTestLayer(name, hc) : NULL;
}

}  // end of namespace PV block

