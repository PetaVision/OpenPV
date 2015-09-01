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

#include <columns/HyPerCol.hpp>
#include <layers/Image.hpp>
#include <layers/Retina.hpp>
#include <connections/HyPerConn.hpp>
#include <layers/ANNLayer.hpp>
#include <io/PointProbe.hpp>
#include <weightinit/InitUniformWeights.hpp>

#include <assert.h>

#undef DEBUG_OUTPUT

// The activity in L1 given a 7x7 weight patch,
// with all weights initialized to 1.
//
#define UNIFORM_ACTIVITY_VALUE 49

using namespace PV;

int check_activity(HyPerLayer * l);

int main(int argc, char * argv[])
{
   int status = 0;

   int rank=0;
   PV_Init* initObj = new PV_Init(&argc, &argv);

   if (pv_getopt(argc, argv, "-p", NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s does not take -p as an option.  Instead the necessary params file is hard-coded.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   int cl_argc = argc+2;
   char ** cl_argv = (char **) calloc((size_t)(cl_argc+1),sizeof(char *));
   assert(cl_argv);
   for (int k=0; k<argc; k++) {
      cl_argv[k] = strdup(argv[k]);
      assert(cl_argv[k]);
   }
   cl_argv[argc] = strdup("-p");
   int paramfile_argnum = argc+1;
   cl_argv[paramfile_argnum] = strdup("input/test_border_activity.params");
   cl_argv[paramfile_argnum+1] = NULL;

   initObj->initialize(cl_argc, cl_argv);
   HyPerCol * hc = new HyPerCol("column", cl_argc, cl_argv, initObj);
   for( int k=0; k<cl_argc; k++ )
   {
      free(cl_argv[k]);
   }
   free(cl_argv); cl_argv=NULL;

   const char * imageLayerName = "test_border_activity image";
   const char * retinaLayerName = "test_border_activity retina";
   const char * l1LayerName = "test_border_activity layer";

   Image * image   = new Image(imageLayerName, hc); assert(image);
   Retina * retina = new Retina(retinaLayerName, hc);           assert(retina);
   ANNLayer * l1     = new ANNLayer(l1LayerName, hc);           assert(l1);

   HyPerConn * conn1 = new HyPerConn("test_border_activity connection 1", hc);
   assert(conn1);
   HyPerConn * conn2 = new HyPerConn("test_border_activity connection 2", hc);
   assert(conn2);
   
#ifdef DEBUG_OUTPUT
   PointProbe * p1 = new PointProbe( 0,  0,  0, "L1 (0,0,0):");
   PointProbe * p2 = new PointProbe(32, 32, 32, "L1 (32,32,0):");
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
