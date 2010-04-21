/*
 * stdp.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: Craig Rasmussen
 */

#include <assert.h>
#include <stdlib.h>

#include "LinearPostConnProbe.hpp"
#include "BiConn.hpp"

#include "Patterns.hpp"

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
   //Image * image = new Gratings("Image", hc);
   Image * image = new Patterns("Bars", hc);

   // create the layers
   //
   HyPerLayer * retina = new Retina("Retina", hc, image);
   HyPerLayer * l1     = new V1("L1", hc);
//   HyPerLayer * l1Inh  = new V1("L1Inh", hc);

   // connect the layers
   //
   HyPerConn * r_l1, * l1_l1Inh, * l1Inh_l1;
//   r_l1     = new BiConn("Retina to L1", hc, retina, l1, CHANNEL_EXC, 0);
   r_l1     = new HyPerConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
//   r_l1     = new RandomConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
//   l1_l1Inh = new HyPerConn( "L1 to L1Inh",  hc, l1,  l1Inh, CHANNEL_EXC);
//   l1Inh_l1 = new HyPerConn("L1Inh to L1",  hc, l1Inh,  l1, CHANNEL_INH);

//   dump_weights(r_l1->weights(0), r_l1->numWeightPatches(0));

#ifdef DISPLAY
   GLDisplay * display = new GLDisplay(&argc, argv, hc, 2, 2);
   display->setDelay(100);
   display->setImage(image);
   display->addLayer(retina);
   display->addLayer(l1);
#endif

   // add probes
   //

//   ConnectionProbe * cProbe0 = new ConnectionProbe(2*5 + 0);
//   PostConnProbe * pcProbe0 = new LinearPostConnProbe(PV::DimX, locY, 0);

#undef LINEAR_PROBES
#ifdef LINEAR_PROBES
   LayerProbe * rProbe0  = new LinearActivityProbe(hc, PV::DimX, 0, 0);
   LayerProbe * rProbe1  = new LinearActivityProbe(hc, PV::DimX, 1, 0);
   LayerProbe * rProbe2  = new LinearActivityProbe(hc, PV::DimX, 2, 0);
   LayerProbe * rProbe3  = new LinearActivityProbe(hc, PV::DimX, 3, 0);
   LayerProbe * rProbe4  = new LinearActivityProbe(hc, PV::DimX, 4, 0);
   LayerProbe * rProbe5  = new LinearActivityProbe(hc, PV::DimX, 5, 0);
   LayerProbe * rProbe6  = new LinearActivityProbe(hc, PV::DimX, 6, 0);
   LayerProbe * rProbe7  = new LinearActivityProbe(hc, PV::DimX, 7, 0);
   LayerProbe * rProbe8  = new LinearActivityProbe(hc, PV::DimX, 8, 0);
   LayerProbe * rProbe9  = new LinearActivityProbe(hc, PV::DimX, 9, 0);
   LayerProbe * rProbe10 = new LinearActivityProbe(hc, PV::DimX, 10, 0);
   LayerProbe * rProbe11 = new LinearActivityProbe(hc, PV::DimX, 11, 0);

   retina->insertProbe(rProbe0);
   retina->insertProbe(rProbe1);
   retina->insertProbe(rProbe2);
   retina->insertProbe(rProbe3);
   retina->insertProbe(rProbe4);
   retina->insertProbe(rProbe5);
   retina->insertProbe(rProbe6);
   retina->insertProbe(rProbe7);
   retina->insertProbe(rProbe8);
   retina->insertProbe(rProbe9);
   retina->insertProbe(rProbe10);
   retina->insertProbe(rProbe11);
#endif

//   LayerProbe * rptprobe = new PointProbe(25, 0, 0, "R :");
//   retina->insertProbe(rptprobe);

//   LayerProbe * ptprobe1 = new PointProbe(53, 3, 0, "L1:");
//   l1->insertProbe(ptprobe1);

   PostConnProbe * pcProbe = new PostConnProbe(778); //(245); // 8575=>127,66
   pcProbe->setImage(image);
   //pcProbe->setOutputIndices(true);
   r_l1->insertProbe(pcProbe);

//   StatsProbe * sProbe = new StatsProbe(PV::BufActivity, "l1");
//   l1->insertProbe(sProbe);

   // run the simulation
   //

   if (hc->columnId() == 0) {
      printf("[0]: Running simulation ...\n");  fflush(stdout);
   }

   hc->run();

   if (hc->columnId() == 0) {
      printf("[0]: Finished\n");
   }

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
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


