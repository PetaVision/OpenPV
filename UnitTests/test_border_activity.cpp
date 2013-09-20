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

#include "../src/columns/HyPerCol.hpp"
#include "../src/layers/Image.hpp"
#include "../src/layers/Retina.hpp"
#include "../src/connections/HyPerConn.hpp"
#include "../src/layers/ANNLayer.hpp"
#include "../src/io/PointProbe.hpp"
#include "../src/weightinit/InitUniformWeights.hpp"

#include <assert.h>

#undef DEBUG_OUTPUT

// The activity in L1 given a 7x7 weight patch,
// with all weights initialized to 1.
//
#define ARGC 3
#define UNIFORM_ACTIVITY_VALUE 49

using namespace PV;

int check_activity(HyPerLayer * l);

int main(int argc, char * argv[])
{
   int status = 0;

   const char * image_file = "input/const_one_64x64.tif";

   char * cl_args[ARGC+1];
   cl_args[0] = strdup(argv[0]);
   cl_args[1] = strdup("-p");
   cl_args[2] = strdup("input/test_border_activity.pv");
   cl_args[3] = NULL;
   HyPerCol * hc = new HyPerCol("column", ARGC, cl_args);
   for( int k=0; k<ARGC; k++ )
   {
      free(cl_args[k]);
   }

   const char * imageLayerName = "test_border_activity image";
   const char * retinaLayerName = "test_border_activity retina";
   const char * l1LayerName = "test_border_activity layer";

   Image * image   = new Image(imageLayerName, hc, image_file); assert(image);
   Retina * retina = new Retina(retinaLayerName, hc);           assert(retina);
   ANNLayer * l1     = new ANNLayer(l1LayerName, hc);           assert(l1);

   InitWeights * weightInit1 = new InitUniformWeights();
   HyPerConn * conn1 = new HyPerConn("test_border_activity connection 1", hc, imageLayerName, retinaLayerName, weightInit1);
   assert(conn1);
   // delete weightInit1; // HyPerConn object takes ownership of InitWeights object
   InitWeights * weightInit2 = new InitUniformWeights();
   HyPerConn * conn2 = new HyPerConn("test_border_activity connection 2", hc, retinaLayerName, l1LayerName, weightInit2);
   assert(conn2);
   
   // delete weightInit2; // HyPerConn object takes ownership of InitWeights object

#ifdef DEBUG_OUTPUT
   PointProbe * p1 = new PointProbe( 0,  0,  0, "L1 (0,0,0):");
   PointProbe * p2 = new PointProbe(32, 32, 32, "L1 (32,32,0):");
   l1->insertProbe(p1);
   l1->insertProbe(p2);
#endif

   // run the simulation
   hc->initFinish();
   hc->run();

   status = check_activity(l1);

   delete hc;
   return status;
}

int check_activity(HyPerLayer * l)
{
   int status = 0;

   const int nx = l->clayer->loc.nx;
   const int ny = l->clayer->loc.ny;
   const int nf = l->clayer->loc.nf;

   const int nk = l->clayer->numNeurons;
   assert(nk == nx*ny*nf);

   for (int k = 0; k < nk; k++) {
      int a = (int) l->clayer->activity->data[k];
      if (a != UNIFORM_ACTIVITY_VALUE) {
         status = -1;
         fprintf(stderr, "ERROR: test_border_activity: activity==%d != %d\n",
                 a, UNIFORM_ACTIVITY_VALUE);
         return status;
      }
   }
   return status;
}
