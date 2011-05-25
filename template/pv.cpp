/*
 * pv.cpp
 *
 *  Created by GJK on May 13th, 6 layer retina model with 3 V layers see
 *
 */

#include <stdlib.h>

#include <src/columns/HyPerCol.hpp>

#include "Patterns.hpp"                     // for getting the bar image pattern from local directory

#include <src/layers/ANNLayer.hpp>          // non firing neurons

#include <src/layers/GapLayer.hpp>          // to get access to V to build gap junctions

#include <src/layers/SigmoidLayer.hpp>      // to get access to V to forward sigmoid

#include <src/layers/LIF.hpp>               // leaky integrate and firing neurons

#include <src/connections/KernelConn.hpp>   // KernelConn, others are derived

#include <src/io/PointProbe.hpp>            // linear activity of a point

#include <src/io/PointLIFProbe.hpp>

#include <src/io/StatsProbe.hpp>

#include <src/layers/Retina.hpp>

using namespace PV;

int main(int argc, char* argv[])
{
   // create the managing hyper column
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the layers
   //
   // main idea:
   //
   // two layers to represent the cones,
   //
   //     an image/pattern followed by ANN, to get
   //     inhibitory channel for the horizontal cells
   //
   // two layers to represent the outer plexiform layer
   //
   //     bipolar cells as ANN
   //     horizontal cell implemented as ann with a
   //       clone layer to do the gap junctions
   //
   // two layers to represent the outer plexiform layer
   //
   //     ganglion cells as LIF (leaky integrate and fire)
   //        with a clone layer to connect the amacrine cells
   //     amacrine cells as LIF with a
   //        clone layer to do the gap junctions
   //

   Image      * Image            = new Patterns("Image", hc, BARS);                         // 0 making bars, code in same directory

   HyPerLayer * Cone             = new LIF("Cone", hc);                                     // 1 to get the inhibitory cone synapes

   HyPerLayer * ConeSigmoid      = new SigmoidLayer("Cone", hc, (LIF*) Cone);               // 2 forward sigmoidal to Bipolar

   HyPerLayer * BipolarON        = new LIF("BipolarON", hc);                                // 3 obvious

   HyPerLayer * BipolarSigmoid   = new SigmoidLayer("BipolarON", hc, (LIF*)BipolarON);      // 4 sigmoidal forward to Ganglion

   HyPerLayer * Horizontal       = new LIF("Horizontal", hc);                               // 5 horizontal with gap junctions and sigmoidal inhibit

   HyPerLayer * HoriGap          = new GapLayer("Horizontal", hc, (LIF*)Horizontal);        // 6 V access for gap junctions

   HyPerLayer * HoriSigmoid      = new SigmoidLayer("Horizontal", hc, (LIF*)Horizontal);    // 7 sigmoidal inhibit for cones

   HyPerLayer * Ganglion         = new LIF("Ganglion", hc);                                 // 8 produce spiking output

   HyPerLayer * GangliGap        = new GapLayer("Ganglion", hc, (LIF*)Ganglion);           // 9 V access for gap junctions

   HyPerLayer * Amacrine         = new LIF("Amacrine", hc);                                // 10 produce spiking output

   HyPerLayer * AmaGap           = new GapLayer("Amacrine", hc, (LIF*) Amacrine);          // 11 V access for gap junctions

   HyPerLayer * RetinaON         = new Retina("RetinaON", hc);                             // debug reference

   // connect the layers - no learning - KernelConn

   new KernelConn("Image to Cone", hc, Image, Cone, CHANNEL_EXC);                               // 0 forward

   new KernelConn("ConeSigmoid to BipolarON", hc, ConeSigmoid, BipolarON, CHANNEL_EXC);         // 1 forward

   new KernelConn("ConeSigmoid to Horizontal", hc, ConeSigmoid, Horizontal, CHANNEL_EXC);       // 2 forward

   new KernelConn("HoriSigmoid to Cone", hc, HoriSigmoid, Cone, CHANNEL_INH);                   // 3 feedback

   new KernelConn("HoriGap to HoriGap", hc, HoriGap, HoriGap, CHANNEL_EXC);                     // 4 gap junction

   new KernelConn("BipolarSigmoid to Ganglion", hc, BipolarSigmoid, Ganglion, CHANNEL_EXC);     // 5 forward

   new KernelConn("BipolarSigmoid to Amacrine", hc, BipolarSigmoid, Amacrine, CHANNEL_EXC);     // 6 forward

   new KernelConn("AmaGap to AmaGap", hc, AmaGap, AmaGap , CHANNEL_EXC);                        // 7 gap junction

   new KernelConn("GangliGap to AmaGap", hc, GangliGap, AmaGap, CHANNEL_EXC);                   // 8 gap junction
   new KernelConn("AmaGap to GangliGap", hc, AmaGap, GangliGap, CHANNEL_EXC);                   // 9 gap junction

   new KernelConn("Amacrine to Ganglion", hc, Amacrine, Ganglion, CHANNEL_INH);                 //10 feedback

   new KernelConn("Image to RetinaON", hc, Image, RetinaON, CHANNEL_EXC);                       // reference



   // add probes

   int locX = 127;
   int locY = 127;      // probing the center
   int locF = 0;        // feature 0

   // remember to delete the probes at the end

   LayerProbe * ptprobeCone = new PointLIFProbe("ptCone.txt",hc,locX, locY, locF, "Cone:");
   Cone->insertProbe(ptprobeCone);

   LayerProbe * ptprobeConeScr = new PointLIFProbe(locX, locY, locF, "Cone:");
   Cone->insertProbe(ptprobeConeScr);

   LayerProbe * statsCone = new StatsProbe("statsCone.txt",hc,BufActivity,"Cone:");
   Cone->insertProbe(statsCone);
//----------------------------------------------------------------------
   LayerProbe * ptprobeBipolarON = new PointLIFProbe("ptBipolarON.txt",hc,locX, locY, locF, "BipolarON:");
     BipolarON->insertProbe(ptprobeBipolarON);
   LayerProbe * ptprobeBipolarONSrc = new PointLIFProbe(locX, locY, locF, "BipolarON:");
     BipolarON->insertProbe(ptprobeBipolarONSrc);

   LayerProbe * statsBipolarON = new StatsProbe("statsBipolarON.txt",hc,BufActivity,"BipolarON:");
   BipolarON->insertProbe(statsBipolarON);
//-----------------------------------------------------------------------
   LayerProbe * ptprobeHorizontal = new PointLIFProbe("ptHorizontal.txt",hc,locX, locY, locF, "Horizontal:");
   Horizontal->insertProbe(ptprobeHorizontal);

   LayerProbe * statsHorizontal = new StatsProbe("statsHorizontal.txt",hc,BufActivity,"Horizontal:");
   Horizontal->insertProbe(statsHorizontal);
//----------------------------------------------------------------------
   LayerProbe * ptprobeGanglion = new PointLIFProbe("ptGanglion.txt",hc,locX, locY, locF, "Ganglion:");
   Ganglion->insertProbe(ptprobeGanglion);
   //LayerProbe * ptprobeGanglionSrc = new PointLIFProbe(locX, locY, locF, "Ganglion:");
   //Ganglion->insertProbe(ptprobeGanglionSrc);

   LayerProbe * statsGanglion = new StatsProbe("statsGanglion.txt",hc,BufActivity,"Ganglion:");
   Ganglion->insertProbe(statsGanglion);
   LayerProbe * statsGanglionScr = new StatsProbe(BufActivity,"Ganglion:");
   Ganglion->insertProbe(statsGanglionScr);

//----------------------------------------------------------------------
   LayerProbe * ptprobeAmacrine = new PointLIFProbe("ptAmacrine.txt",hc,locX, locY, locF, "Amacrine:");
   Amacrine->insertProbe(ptprobeAmacrine);

   LayerProbe * statsAmacrine = new StatsProbe("statsAmacrine.txt",hc,BufActivity,"Amacrine:");
   Amacrine->insertProbe(statsAmacrine);

   //----------------------------------------------------------------------
   LayerProbe * ptprobeRetinaON = new PointProbe("ptRetinaON.txt",hc,locX, locY, locF, "RetinaON:");
   RetinaON->insertProbe(ptprobeRetinaON);

   LayerProbe * statsRetinaON = new StatsProbe("statsRetinaON.txt",hc,BufActivity,"RetinaON:");
   RetinaON->insertProbe(statsRetinaON);

   LayerProbe * statsRetinaONSrc = new StatsProbe(BufActivity,"RetinaON:");
   RetinaON->insertProbe(statsRetinaONSrc);



	// run the simulation
   hc->initFinish();

   if (hc->columnId() == 0) {
      printf("[0]: Running simulation ...\n");  fflush(stdout);
   }

   hc->run();

   if (hc->columnId() == 0) {
      printf("[0]: Finished\n");  fflush(stdout);
   }

   // clean up (HyPerCol owns layers and connections, don't delete them)
   delete hc;
   //delete probes;
   delete ptprobeCone;
   delete statsCone;

   delete ptprobeBipolarON;
   delete statsBipolarON;

   delete ptprobeHorizontal;
   delete statsHorizontal;

   delete ptprobeGanglion;
   delete statsGanglion;

   delete ptprobeAmacrine;
   delete statsAmacrine;

   delete ptprobeRetinaON;
   delete statsRetinaON;



   return 0;
}
