/**
 * test_border_activity.cpp
 *
 *  Created on: Feb 11, 2010
 *      Author: Craig Rasmussen
 *
 * This file tests activity near extended borders.  With mirror boundary
 * conditions a uniform retinal input should produce uniform output across
 * the post-synaptic layer.
 */

#include "cMakeHeader.h"
#include "columns/HyPerCol.hpp"
#include "connections/HyPerConn.hpp"
#include "layers/ANNLayer.hpp"
#include "layers/PvpLayer.hpp"
#include "layers/Retina.hpp"
#include "probes/PointProbe.hpp"
#include "utils/PVLog.hpp"
#include "weightinit/InitUniformWeights.hpp"

#undef DEBUG_OUTPUT

// The activity in L1 given a 7x7 weight patch,
// with all weights initialized to 1.
//
#define UNIFORM_ACTIVITY_VALUE 49

using namespace PV;

int check_activity(HyPerLayer *l);

int main(int argc, char *argv[]) {

   int status = 0;

   int rank         = 0;
   PV_Init *initObj = new PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);

   if (initObj->getParams() != nullptr) {
      if (rank == 0) {
         ErrorLog().printf(
               "%s does not take -p as an option.  Instead the necessary params file is "
               "hard-coded.\n",
               argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   initObj->setParams("input/test_border_activity.params");

   HyPerCol *hc = new HyPerCol("column", initObj);

   const char *imageLayerName  = "test_border_activity_image";
   const char *retinaLayerName = "test_border_activity_retina";
   const char *l1LayerName     = "test_border_activity_layer";

   PvpLayer *image = new PvpLayer(imageLayerName, hc);
   assert(image);
   Retina *retina = new Retina(retinaLayerName, hc);
   assert(retina);
   ANNLayer *l1 = new ANNLayer(l1LayerName, hc);
   assert(l1);

   HyPerConn *conn1 = new HyPerConn("test_border_activity_connection1", hc);
   FatalIf(!(conn1), "Test failed.\n");
   HyPerConn *conn2 = new HyPerConn("test_border_activity_connection2", hc);
   FatalIf(!(conn2), "Test failed.\n");

#ifdef DEBUG_OUTPUT
   PointProbe *p1 = new PointProbe(0, 0, 0, "L1 (0,0,0):");
   PointProbe *p2 = new PointProbe(32, 32, 32, "L1 (32,32,0):");
   l1->insertProbe(p1);
   l1->insertProbe(p2);
#endif

   // run the simulation
   hc->run();

   status = check_activity(l1);

   delete hc;

   delete initObj;

   return status;
}

int check_activity(HyPerLayer *l) {
   int status = 0;

   const int nx = l->clayer->loc.nx;
   const int ny = l->clayer->loc.ny;
   const int nf = l->clayer->loc.nf;

   const int nk = l->clayer->numNeurons;
   FatalIf(!(nk == nx * ny * nf), "Test failed.\n");

   for (int k = 0; k < nk; k++) {
      int a = (int)l->clayer->activity->data[k];
      if (a != UNIFORM_ACTIVITY_VALUE) {
         status = -1;
         ErrorLog().printf("test_border_activity: activity==%d != %d\n", a, UNIFORM_ACTIVITY_VALUE);
         return status;
      }
   }
   return status;
}
