/*
 * SparsityLayerProbe.cpp
 *
 *  Created on: Apr 2, 2014
 *      Author: slundquist
 */

#include "SparsityLayerProbe.hpp"
#include "../layers/HyPerLayer.hpp"

namespace PV {

/**
 * @filename
 */
SparsityLayerProbe::SparsityLayerProbe(const char * probeName, HyPerCol * hc)
{
   initSparsityLayerProbe_base();
   LayerProbe::initLayerProbe(probeName, hc);
}

SparsityLayerProbe::SparsityLayerProbe()
   : LayerProbe()
{
   initSparsityLayerProbe_base();
   // Derived classes should call initStatsProbe
}

int SparsityLayerProbe::initSparsityLayerProbe_base() {
   sparsityVal = 0;
}

/**
 * @time
 */
int SparsityLayerProbe::outputState(double timef)
{
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   MPI_Comm comm = icComm->communicator();
   int rank = icComm->commRank();
   const int rcvProc = 0;
#endif // PV_USE_MPI
   const pvdata_t * buf = getTargetLayer()->getLayerData();
   int nk = getTargetLayer()->getNumNeurons();
   int nnz = 0;
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   for( int k=0; k<nk; k++ ) {
      int kex = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
      pvdata_t a = buf[kex];
      if(a > 0){
         nnz++;
      }
   }
#ifdef PV_USE_MPI
   //Sum all nnz across processors
   MPI_Allreduce(MPI_IN_PLACE, &nnz, 1, MPI_INT, MPI_SUM, comm);
#endif // PV_USE_MPI
   int numTotNeurons = loc->nxGlobal * loc->nyGlobal * loc->nf;
   sparsityVal = (float)nnz/numTotNeurons;
   return PV_SUCCESS;
}

} // namespace PV
