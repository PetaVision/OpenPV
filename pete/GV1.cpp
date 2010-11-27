/*
 * GV1.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 *      Author: pschultz
 *
 *  Overrides recvSynapticInput to include negative weights
 */

#include "GV1.hpp"

namespace PV {

GV1::GV1(const char* name, HyPerCol * hc) : V1(name, hc) {
}

GV1::GV1(const char* name, HyPerCol * hc, PVLayerType type) : V1(name, hc, type) {
}

int GV1::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity, int neighbor)
{
   assert(neighbor >= 0);
   const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   fflush(stdout);
#endif

   for (int kPre = 0; kPre < numExtended; kPre++) {
      float a = activity->data[kPre];
      if (a == 0.0f) continue;  // TODO - assume activity is sparse so make this common branch
      //    ^^ This is the only change from HyPerLayer::recvSynapticInput

      PVAxonalArbor * arbor = conn->axonalArbor(kPre, neighbor);
      PVPatch * phi = arbor->data;
      PVPatch * weights = arbor->weights;

      // WARNING - assumes weight and phi patches from task same size
      //         - assumes patch stride sf is 1

      int nk  = phi->nf * phi->nx;
      int ny  = phi->ny;
      int sy  = phi->sy;        // stride in layer
      int syw = weights->sy;    // stride in patch

      // TODO - unroll
      for (int y = 0; y < ny; y++) {
         pvpatch_accumulate(nk, phi->data + y*sy, a, weights->data + y*syw);
//       if (err != 0) printf("  ERROR kPre = %d\n", kPre);
      }
   }

   return 0;
}


}  // end namespace PV
