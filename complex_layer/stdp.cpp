/*
 * stdp.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: Craig Rasmussen
 */

#include <assert.h>
#include <stdlib.h>


#include "Patterns.hpp"

#include <src/columns/HyPerCol.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/GLDisplay.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/PointLIFProbe.hpp>
#include <src/io/StatsProbe.hpp>
#include <src/layers/Gratings.hpp>
#include <src/layers/Movie.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/LIF.hpp>
#include <src/connections/HyPerConn.hpp>
#include <src/connections/RandomConn.hpp>
#include <src/connections/KernelConn.hpp>

using namespace PV;

#undef DISPLAY

void dump_weights(PVPatch ** patches, int numPatches);

#define INHIB

#undef C1

#undef C1INHIB

#undef S2

#undef S2INHIB

#undef S2toC1



int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   //
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the layers
   //

   Image * image = new Movie("Image", hc, "../t_fileOfFilenames.txt", 10.0);
   //Image * image = new Patterns("Image", hc, RECTANGLES);
   //Image * image = new Patterns("Image", hc, BARS);

   HyPerLayer * retinaOn  = new Retina("RetinaOn", hc);
   HyPerLayer * retinaOff = new Retina("RetinaOff", hc);
   HyPerLayer * s1        = new LIF("S1", hc);

#ifdef INHIB
   HyPerLayer * s1Inh = new LIF("S1Inh", hc);
#endif

#ifdef C1
   HyPerLayer * c1 = new LIF("C1", hc);
#endif

#ifdef C1INHIB
   HyPerLayer * c1Inh = new LIF("C1Inh", hc);
#endif

#ifdef S2
   HyPerLayer * s2 = new LIF("S2", hc);
#endif

#ifdef S2INHIB
   HyPerLayer * s2Inh = new LIF("S2Inh", hc);
#endif


   // connect the layers
   //

   HyPerConn * i_r1_c  = new KernelConn("Image to RetinaOn Center",   hc, image, retinaOn, CHANNEL_EXC);
   HyPerConn * i_r1_s  = new KernelConn("Image to RetinaOn Surround", hc, image, retinaOn, CHANNEL_INH);
   HyPerConn * i_r0_c  = new KernelConn("Image to RetinaOff Center", hc, image, retinaOff, CHANNEL_INH);
   HyPerConn * i_r0_s  = new KernelConn("Image to RetinaOff Surround", hc, image, retinaOff, CHANNEL_EXC);
   HyPerConn * r1_s1   = new HyPerConn("RetinaOn to S1", hc, retinaOn, s1, CHANNEL_EXC);
   HyPerConn * r0_s1   = new HyPerConn("RetinaOff to S1", hc, retinaOff, s1, CHANNEL_EXC);
#ifdef C1
   HyPerConn * s1_c1   = new HyPerConn("S1 to C1", hc, s1, c1, CHANNEL_EXC);
#endif

#ifdef C1INHIB
   HyPerConn * c1_c1Inh = new KernelConn("C1 to C1Inh", hc, c1, c1Inh, CHANNEL_EXC);
   HyPerConn * c1Inh_c1 = new KernelConn("C1Inh to C1", hc, c1Inh, c1, CHANNEL_INH);
#endif

#ifdef INHIB
   HyPerConn * s1_s1Inh = new KernelConn("S1 to S1Inh",  hc, s1,  s1Inh, CHANNEL_EXC);
   HyPerConn * s1Inh_s1 = new KernelConn("S1Inh to S1",  hc, s1Inh,  s1, CHANNEL_INH);
#endif

#ifdef S2
   HyPerConn * c1_s2   = new HyPerConn("C1 to S2", hc, c1, s2, CHANNEL_EXC);
#endif

#ifdef S2INHIB
   HyPerConn * s2_s2Inh = new KernelConn("S2 to S2Inh", hc, s2, s2Inh, CHANNEL_EXC);
   HyPerConn * s2Inh_s2 = new KernelConn("S2Inh to S2", hc, s2Inh, s2, CHANNEL_INH);
#endif

#ifdef S2toC1
   HyPerConn * s2_c1   = new HyPerConn("S2 to C1", hc, s2, c1, CHANNEL_EXC);
#endif



#ifdef DISPLAY
   GLDisplay * display = new GLDisplay(&argc, argv, hc, 2, 2);
   display->setDelay(800);
   display->setImage(image);
   display->addLayer(retinaOn);
   display->addLayer(retinaOff);
   display->addLayer(s1);
#endif

   // add probes
   //

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
   LayerProbe * rProbe10  = new LinearActivityProbe(hc, PV::DimX, 10, 0);
   LayerProbe * rProbe11  = new LinearActivityProbe(hc, PV::DimX, 11, 0);
   LayerProbe * rProbe12  = new LinearActivityProbe(hc, PV::DimX, 12, 0);
   LayerProbe * rProbe13  = new LinearActivityProbe(hc, PV::DimX, 13, 0);
   LayerProbe * rProbe14  = new LinearActivityProbe(hc, PV::DimX, 14, 0);

   l2->insertProbe(rProbe0);
   l2->insertProbe(rProbe1);
   l2->insertProbe(rProbe2);
   l2->insertProbe(rProbe3);
   l2->insertProbe(rProbe4);
   l2->insertProbe(rProbe5);
   l2->insertProbe(rProbe6);
   l2->insertProbe(rProbe7);
   l2->insertProbe(rProbe8);
   l2->insertProbe(rProbe9);
   l2->insertProbe(rProbe10);
   l2->insertProbe(rProbe11);
   l2->insertProbe(rProbe12);
   l2->insertProbe(rProbe13);
   l2->insertProbe(rProbe14);

#endif

#undef WRITE_KERNELS
#ifdef WRITE_KERNELS

	const char * i_r1_s_filename = "i_r1_s_gauss.txt";
	HyPerLayer * pre = i_r1_s->preSynapticLayer();
	int npad = pre->clayer->loc.nb;
	int nx = pre->clayer->loc.nx;
	int ny = pre->clayer->loc.ny;
	int nf = pre->clayer->loc.nf;
	i_r1_s->writeTextWeights(i_r1_s_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * i_r0_s_filename = "i_r0_s_gauss.txt";
	pre = i_r0_s->preSynapticLayer();
	npad = pre->clayer->loc.nb;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nf;
	i_r0_s->writeTextWeights(i_r0_s_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

#endif
   //40
   //LayerProbe * ptprobe1 = new PointProbe("l2_activity.txt", 9, 7, 0, "L2:");
   //l2->insertProbe(ptprobe1);

   //ConnectionProbe * cProbe = new ConnectionProbe(114);
   //l1_l2->insertProbe(cProbe);

   //ConnectionProbe * cProbe = new ConnectionProbe(277);
   //i_r1_s->insertProbe(cProbe);

   //PostConnProbe * pcProbe0 = new LinearPostConnProbe(PV::DimX, locY, 0);

   //LayerProbe * rptprobe = new PointProbe(25, 0, 0, "R :");
   //retina->insertProbe(rptprobe);

   //LayerProbe * ptprobe1 = new PointLIFProbe("l2_activity.txt", 10, 38, 0, "L2:");
   //l2->insertProbe(ptprobe1);

   //LayerProbe * ptprobe2 = new PointLIFProbe("l2Inh_activity.txt", 5, 16, 0, "L2Inh:");
   //l2Inh->insertProbe(ptprobe2);

   //LayerProbe * ptprobe3 = new PointProbe("image_activity.txt", 17, 11, 0, "Image:");
   //image->insertProbe(ptprobe3);

   //int retinaplacement = 9;

   //LayerProbe * ptprobe4 = new PointProbe("retinaOn_activity.txt", retinaplacement, 11, 0, "RetinaOn:");
   //retinaOn->insertProbe(ptprobe4);

   //LayerProbe * ptprobe5 = new PointProbe("retinaOff_activity.txt", retinaplacement, 11, 0, "RetinaOff:");
   //retinaOff->insertProbe(ptprobe5);

   //LayerProbe * ptprobe6 = new PointProbe("l1_activity.txt", 52, 40, 0, "L1:");
   //l1->insertProbe(ptprobe6);

   //LayerProbe * ptprobe7 = new PointProbe("l1Inh_activity.txt", 13, 10, 0, "L1Inh:");
   //l1Inh->insertProbe(ptprobe7);

   //PostConnProbe * pcOnProbe  = new PostConnProbe(5474); //(245); // 8575=>127,66
   //PostConnProbe * pcOffProbe = new PostConnProbe(5474); //(245); // 8575=>127,66
   //PostConnProbe * pcInhProbe = new PostConnProbe(403);
   //pcProbe->setImage(image);
   //pcOnProbe->setOutputIndices(true);
   //r0_l1->insertProbe(pcOffProbe);
   //r1_l1->insertProbe(pcOnProbe);
   //l1_l1Inh->insertProbe(pcInhProbe);

   //StatsProbe * sProbe = new StatsProbe(PV::BufActivity, "l1");
   //l1->insertProbe(sProbe);

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



