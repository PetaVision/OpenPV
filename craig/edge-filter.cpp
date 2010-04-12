/*
 * stdp.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: Craig Rasmussen
 */

#include <stdlib.h>

#include "LinearPostConnProbe.hpp"
#include "BiConn.hpp"

#include <src/columns/HyPerCol.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/GLDisplay.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/StatsProbe.hpp>
#include <src/layers/Gratings.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/connections/HyPerConn.hpp>
#include <src/connections/RandomConn.hpp>

using namespace PV;

#undef DISPLAY

void dump_weights(PVPatch ** patches, int numPatches);

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   //
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the image
   //
   Image * image = new Gratings("Image", hc);

   // create the layers
   //
   HyPerLayer * retina = new Retina("Retina", hc, image);
   HyPerLayer * l1     = new V1("L1", hc);

   // connect the layers
   //
   HyPerConn * r_l1_c, * r_l1_s;
   r_l1_c = new HyPerConn("Retina to L1 Center",   hc, retina, l1, CHANNEL_EXC);
   r_l1_s = new HyPerConn("Retina to L1 Surround", hc, retina, l1, CHANNEL_EXC);

//   dump_weights(r_l1->weights(0), r_l1->numWeightPatches(0));

   // add probes
   //

   PostConnProbe * pcProbe_c = new PostConnProbe(8575); // 127,66
   PostConnProbe * pcProbe_s = new PostConnProbe(8575); // 127,66

   pcProbe_c->setOutputIndices(true);
   r_l1_c->insertProbe(pcProbe_c);

   pcProbe_s->setOutputIndices(true);
   r_l1_s->insertProbe(pcProbe_s);

   // run the simulation
   //
   hc->run(1);

   // clean up (HyPerCol owns layers and connections, don't delete them)
   //
   delete hc;

   return 0;
}

void dump_weights(PVPatch ** patches, int numPatches)
{
   FILE * fp = fopen("wdump.txt", "w");
   assert(fp != NULL);

   int num = 0;
   for (int k = 0; k < numPatches; k++) {
      PVPatch * p = patches[k];
      float * w = p->data;
      int nkp = p->nx * p->ny * p->nf;
      for (int i = 0; i < nkp; i++) {
         fprintf(fp, "%d (%d,%d)  %f\n", num, k, i, w[i]);
         num += 1;
      }
   }
   fclose(fp);
}


