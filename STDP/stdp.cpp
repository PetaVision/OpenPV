/*
 * pv_ca.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

#include <src/columns/HyPerCol.hpp>
#include <src/connections/RandomConn.hpp>
#include <src/layers/Image.hpp>
#include <src/layers/Movie.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>

using namespace PV;

int main(int argc, char* argv[])
{
   const char * files = "input/movies/movies.files";

   // create the managing hypercolumn
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // read an image
   float displayPeriod = 20;  // ms
   Movie * movie = new Movie(files, hc, displayPeriod);

   // create the layers
   HyPerLayer * retina = new Retina("Retina", hc, movie);
   HyPerLayer * l1     = new V1("L1", hc);

   // connect the layers
   HyPerConn * r_l1 = new RandomConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);

   // add probes

//   PVLayerProbe * ptprobeI = new PointProbe(61, locY, 0, "LI:x=61 f=0");

//   ConnectionProbe * cProbe0 = new ConnectionProbe(2*5 + 0);
//   ConnectionProbe * cProbe0 = new PostConnProbe(0 + 5*nfPost);

//   retina->insertProbe(rProbe0);
//   retina->insertProbe(rProbe1);
   //r_l1->insertProbe(cProbe);

//   l1->insertProbe(probe0);

//   r_l1->insertProbe(cProbe0);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete movie;
   delete hc;

   return 0;
}
