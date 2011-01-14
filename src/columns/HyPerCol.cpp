/*
 * HyPerCol.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#define TIMER_ON
#undef TIMESTEP_OUTPUT

#include "HyPerCol.hpp"
#include "InterColComm.hpp"
#include "../connections/PVConnection.h"
#include "../io/clock.h"
#include "../io/imageio.hpp"
#include "../io/io.h"
#include "../utils/pv_random.h"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

namespace PV {

HyPerCol::HyPerCol(const char * name, int argc, char * argv[])
         : warmStart(false), isInitialized(false)
{
   initialize(name, argc, argv);
}

HyPerCol::HyPerCol(const char * name, int argc, char * argv[], const char * pv_path)
         : warmStart(false), isInitialized(false)
{
   initialize(name, argc, argv);
   assert(strlen(path) + strlen(pv_path) < PV_PATH_MAX);
   sprintf(path, "%s/%s", path, pv_path);

}

HyPerCol::~HyPerCol()
{
   int n;

   finalizeThreads();

   if (image_file != NULL) free(image_file);

   for (n = 0; n < numConnections; n++) {
      delete connections[n];
   }

   for (n = 0; n < numLayers; n++) {
      // TODO: check to see if finalize called
      if (layers[n] != NULL) {
         delete layers[n]; // will call *_finalize
      }
      else {
         // TODO move finalize
         // PVLayer_finalize(getCLayer(n));
      }
   }

   delete icComm;

   printf("%32s: total time in %6s %10s: ", name, "column", "run    ");
   runTimer->elapsed_time();
   fflush(stdout);
   delete runTimer;

   free(connections);
   free(layers);
   free(name);
   free(path);
}

int HyPerCol::initFinish(void)
{
   int status = 0;

   for (int i = 0; i < this->numLayers; i++) {
      status = layers[i]->initFinish();
      if (status != 0) {
         fprintf(stderr, "[%d]: HyPerCol::initFinish: ERROR condition, exiting...\n", this->columnId());
         exit(status);
      }
   }

#ifdef OBSOLETE
   // TODO - fix this to modern version?
   log_parameters(numSteps, image_file);
#endif

   isInitialized = true;

   return status;
}

int HyPerCol::initialize(const char * name, int argc, char ** argv)
{

   int opencl_device = 1;  // default to CPU for now

   layerArraySize = INITIAL_LAYER_ARRAY_SIZE;
   connectionArraySize = INITIAL_CONNECTION_ARRAY_SIZE;

   // maxLayers = MAX_LAYERS;
   // maxConnections = MAX_CONNECTIONS;

   this->name = strdup(name);
   this->runTimer = new Timer();

   path = (char *) malloc(1+PV_PATH_MAX);
   assert(path != NULL);
   path = getcwd(path, PV_PATH_MAX);

   char * param_file;
   simTime = 0;
   numLayers = 0;
   numConnections = 0;
   layers = (HyPerLayer **) malloc(layerArraySize * sizeof(HyPerLayer *));
   connections = (HyPerConn **) malloc(connectionArraySize * sizeof(HyPerConn *));

   numSteps = 2;
   image_file = NULL;
   param_file = NULL;
   unsigned long random_seed = 0;
   parse_options(argc, argv, &image_file, &param_file, &numSteps, &opencl_device, &random_seed);

   // run only on CPU for now
   initializeThreads(opencl_device);

   // estimate for now
   // TODO -get rid of maxGroups
   int maxGroups = 2*(layerArraySize + connectionArraySize);
   params = new PVParams(param_file, maxGroups);

   icComm = new InterColComm(&argc, &argv);

   // initialize random seed
   //
   random_seed = getRandomSeed();
   random_seed = params->value(name, "randomSeed", random_seed);
   pv_srandom(random_seed);

   if (param_file != NULL) free(param_file);

   deltaTime = DELTA_T;
   if (params->present(name, "dt")) deltaTime = params->value(name, "dt");

   int status = -1;
   if (image_file) {
      status = getImageInfo(image_file, icComm, &imageLoc);
   }

   if (status) {
      imageLoc.nxGlobal = (int) params->value(name, "nx");
      imageLoc.nyGlobal = (int) params->value(name, "ny");

      // set loc based on global parameters and processor partitioning
      //
      setLayerLoc(&imageLoc, 1.0f, 1.0f, 0, 1);
   }

   runDelegate = NULL;

   numProbes = 0;
   probes = NULL;

   return EXIT_SUCCESS;
}

int HyPerCol::columnId()
{
   return icComm->commRank();
}

int HyPerCol::numberOfColumns()
{
   return icComm->numCommRows() * icComm->numCommColumns();
}

int HyPerCol::commColumn(int colId)
{
   return colId % icComm->numCommColumns();
}

int HyPerCol::commRow(int colId)
{
   return colId / icComm->numCommColumns();
}

int HyPerCol::setLayerLoc(PVLayerLoc * layerLoc,
                          float nxScale, float nyScale, int margin, int nf)
{
   layerLoc->nxGlobal = (int) (nxScale * imageLoc.nxGlobal);
   layerLoc->nyGlobal = (int) (nyScale * imageLoc.nyGlobal);

   // partition input space based on the number of processor
   // columns and rows
   //

   layerLoc->nx = layerLoc->nxGlobal / icComm->numCommColumns();
   layerLoc->ny = layerLoc->nyGlobal / icComm->numCommRows();

   assert(layerLoc->nxGlobal == layerLoc->nx * icComm->numCommColumns());
   assert(layerLoc->nyGlobal == layerLoc->ny * icComm->numCommRows());

   layerLoc->kx0 = layerLoc->nx * icComm->commColumn();
   layerLoc->ky0 = layerLoc->ny * icComm->commRow();

   layerLoc->nf = nf;
   layerLoc->nb = margin;

   layerLoc->halo.lt = margin;
   layerLoc->halo.rt = margin;
   layerLoc->halo.dn = margin;
   layerLoc->halo.up = margin;

   return 0;
}

int HyPerCol::addLayer(HyPerLayer * l)
{
   assert(numLayers <= layerArraySize);
   if( numLayers ==  layerArraySize ) {
      layerArraySize += RESIZE_ARRAY_INCR;
      HyPerLayer ** newLayers = (HyPerLayer **) malloc( layerArraySize * sizeof(HyPerLayer *) );
      assert(newLayers);
      for(int k=0; k<numLayers; k++) {
         newLayers[k] = layers[k];
      }
      free(layers);
      layers = newLayers;
   }
   l->columnWillAddLayer(icComm, numLayers);
   layers[numLayers++] = l;
   return (numLayers - 1);
}

int HyPerCol::addConnection(HyPerConn * conn)
{
   int connId = numConnections;

   assert(numConnections <= connectionArraySize);
   if( numConnections == connectionArraySize ) {
      connectionArraySize += RESIZE_ARRAY_INCR;
      HyPerConn ** newConnections = (HyPerConn **) malloc( connectionArraySize * sizeof(HyPerConn *) );
      assert(newConnections);
      for(int k=0; k<numConnections; k++) {
         newConnections[k] = connections[k];
      }
      free(connections);
      connections = newConnections;
   }

   // numConnections is the ID of this connection
   icComm->subscribe(conn);

   connections[numConnections++] = conn;

   return connId;
}

int HyPerCol::run(int nTimeSteps)
{
   checkMarginWidths();

   int step = 0;
   float stopTime = simTime + nTimeSteps * deltaTime;
   const bool exitOnFinish = false;

   if (!isInitialized) {
      initFinish();
   }

   numSteps = nTimeSteps;

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol: running...\n");  fflush(stdout);
   }
#endif

   // publish initial conditions
   //
   for (int l = 0; l < numLayers; l++) {
      layers[l]->publish(icComm, simTime);
   }

   // wait for all published data to arrive
   //
   for (int l = 0; l < numLayers; l++) {
      icComm->wait(layers[l]->getLayerId());
   }

   if (runDelegate) {
      // let delegate advance the time
      //
      runDelegate->run(simTime, stopTime);
   }

#ifdef TIMER_ON
      start_clock();
#endif
   // time loop
   //
   while (simTime < stopTime) {
      simTime = advanceTime(simTime);
      step += 1;

#ifdef TIMER_ON
      if (step == 10) start_clock();
#endif

   }  // end time loop

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol::run done...\n");  fflush(stdout);
   }
#endif

   exitRunLoop(exitOnFinish);

#ifdef TIMER_ON
      stop_clock();
#endif

   return 0;
}

float HyPerCol::advanceTime(float sim_time)
{
#ifdef TIMESTEP_OUTPUT
   int nstep = (int) (sim_time/getDeltaTime());
   if (nstep%2000 == 0 && columnId() == 0) {
      printf("   [%d]: time==%f\n", columnId(), sim_time);
   }
#endif

   runTimer->start();

   // At this point all activity from the previous time step have
   // been delivered to the data store.
   //

   // update the connections (weights)
   //
   for (int c = 0; c < numConnections; c++) {
      connections[c]->updateState(sim_time, deltaTime);
      connections[c]->outputState(sim_time);
   }

   // Update the layers (activity)
   //
   for (int l = 0; l < numLayers; l++) {
      layers[l]->outputState(sim_time);

      // deliver new synaptic activity to layer
      //
      layers[l]->triggerReceive(icComm);

      // update layer and calculate new activity
      //
      layers[l]->updateState(sim_time, deltaTime);
   }

   // This loop separate from the update layer loop above
   // to provide time for layer data to be copied from
   // the OpenCL device.
   //
   for (int l = 0; l < numLayers; l++) {
      // after updateBorder completes all necessary data has been
      // copied from the device (GPU) to the host (CPU)
      layers[l]->updateBorder(sim_time, deltaTime);

      // TODO - move this to layer
      // Advance time level so we have a new place in data store
      // to copy the data.  This should be done immediately before
      // publish so there is a place to publish and deliver the data to.
      // No one can access the data store (except to publish) until
      // wait has been called.  This should be fixed so that publish goes
      // to last time level and level is advanced only after wait.
      icComm->increaseTimeLevel(layers[l]->getLayerId());

      layers[l]->publish(icComm, sim_time);
   }

   // wait for all published data to arrive
   //
   for (int l = 0; l < numLayers; l++) {
      layers[l]->waitOnPublish(icComm);
   }

   // make sure simTime is updated even if HyPerCol isn't running time loop

   float outputTime = simTime; // so that outputState is called with the correct time
                               // but doesn't effect runTimer

   simTime = sim_time + deltaTime;

   runTimer->stop();

   outputState(outputTime);

   return simTime;
}

int HyPerCol::exitRunLoop(bool exitOnFinish)
{
   int status = 0;

   // output final state of layers and connections
   //
   bool last = true;

   for (int l = 0; l < numLayers; l++) {
      layers[l]->writeState(layers[l]->getName(), simTime, last);
   }

   for (int c = 0; c < numConnections; c++) {
      connections[c]->outputState(simTime, last);
   }

   if (exitOnFinish) {
      delete this;
      exit(0);
   }

   return status;
}

int HyPerCol::initializeThreads(int device)
{
   clDevice = new CLDevice(device);
   return 0;
}

int HyPerCol::finalizeThreads()
{
   delete clDevice;
   return 0;
}

int HyPerCol::loadState()
{
   return 0;
}

int HyPerCol::writeState()
{
   for (int l = 0; l < numLayers; l++) {
      layers[l]->writeState(OUTPUT_PATH, simTime);
   }
   return 0;
}

int HyPerCol::insertProbe(ColProbe * p)
{
   ColProbe ** newprobes;
   newprobes = (ColProbe **) malloc((numProbes + 1) * sizeof(ColProbe *));
   assert(newprobes != NULL);

   for (int i = 0; i < numProbes; i++) {
      newprobes[i] = probes[i];
   }
   delete probes;

   probes = newprobes;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerCol::outputState(float time)
{
   for( int n = 0; n < numProbes; n++ ) {
       probes[n]->outputState(time, this);
   }
   return EXIT_SUCCESS;
}

int HyPerCol::checkMarginWidths() {
   // For each connection, make sure that the post-synaptic margin width is
   // large enough for the patch size.

   // TODO instead of having marginWidth supplied to HyPerLayers in the
   // params.pv file, calculate them based on the patch sizes here.
   // Hard part:  numExtended-sized quantities (e.g. clayer->activity) can't
   // be allocated and initialized until after nPad is determined.

   int status = EXIT_SUCCESS;
   int status1, status2;
   for( int c=0; c < numConnections; c++ ) {
      HyPerConn * conn = connections[c];
      int padding = conn->pre->getLayerLoc()->nb;

      int xScalePre = conn->preSynapticLayer()->getXScale();
      int xScalePost = conn->postSynapticLayer()->getXScale();
      status1 = zCheckMarginWidth(conn, "x", padding, conn->xPatchSize(), xScalePre, xScalePost, status);

      int yScalePre = conn->preSynapticLayer()->getYScale();
      int yScalePost = conn->postSynapticLayer()->getYScale();
      status2 = zCheckMarginWidth(conn, "y", padding, conn->yPatchSize(), yScalePre, yScalePost, status);
      status = (status == EXIT_SUCCESS && status1 == EXIT_SUCCESS && status2 == EXIT_SUCCESS) ?
               EXIT_SUCCESS : EXIT_FAILURE;
   }
   return status;
}  // end HyPerCol::checkMarginWidths()

int HyPerCol::zCheckMarginWidth(HyPerConn * conn, const char * dim, int padding, int patchSize, int scalePre, int scalePost, int prevStatus) {
   int status;
   int scaleDiff = scalePre - scalePost;
   // if post has higher neuronal density than pre, scaleDiff < 0.
   int needed = scaleDiff > 0 ? ( patchSize/( (int) powf(2,scaleDiff) )/2 ) :
                                ( (patchSize/2) * ( (int) powf(2,-scaleDiff) ) );
   if( padding < needed ) {
      if( prevStatus == EXIT_SUCCESS ) {
         fprintf(stderr, "Margin width error.\n");
      }
      fprintf(stderr, "Connection \"%s\", dimension %s:\n", conn->getName(), dim);
      fprintf(stderr, "    Margin width %d, patch size %d, presynaptic scale %d, postsynaptic scale %d\n",
              padding, patchSize, scalePre, scalePost);
      fprintf(stderr, "    Needed margin width=%d\n", needed);
      status = EXIT_FAILURE;
      if( numberOfColumns() > 1 || padding > 0 ) {
         fprintf(stderr, "Exiting.\n");
         exit(EXIT_FAILURE);
      }
      else {
         fprintf(stderr, "Continuing, but there may be undesirable edge effects.\n");
      }
   }
   else status = EXIT_SUCCESS;
   return status;
}

} // PV namespace
