/*
 * HyPerCol.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#undef TIMER_ON
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

namespace PV {

HyPerCol::HyPerCol(const char * name, int argc, char * argv[])
         : warmStart(false), isInitialized(false)
{
   // TODO - fix these numbers to dynamically grow
   maxLayers = MAX_LAYERS;
   maxConnections = MAX_CONNECTIONS;

   this->name = strdup(name);

   char * param_file;
   simTime = 0;
   numLayers = 0;
   numConnections = 0;
   layers = (HyPerLayer **) malloc(maxLayers * sizeof(HyPerLayer *));
   connections = (HyPerConn **) malloc(maxConnections * sizeof(HyPerConn *));

#ifdef MULTITHREADED
   numThreads = 1;
#else
   numThreads = 0;
#endif

   numSteps = 2;
   image_file = NULL;
   param_file = NULL;
   parse_options(argc, argv, &image_file, &param_file, &numSteps, &numThreads);

   // estimate for now
   // TODO -get rid of maxGroups
   int maxGroups = 2*(maxLayers + maxConnections);
   params = new PVParams(param_file, maxGroups);

   icComm = new InterColComm(&argc, &argv);

   // initialize random seed
   //
   pv_srandom(getRandomSeed());

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
}

HyPerCol::~HyPerCol()
{
   int n;

#ifdef MULTITHREADED
   finalizeThreads();
#endif

   if (image_file != NULL) free(image_file);

   for (n = 0; n < numConnections; n++) {
      delete connections[n];
   }

   free(threadCLayers);

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

   free(connections);
   free(layers);
   free(name);
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

   log_parameters(numSteps, image_file);

#ifdef MULTITHREADED
   initializeThreads();
#endif

   isInitialized = true;

   return status;
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

   layerLoc->nPad = margin;
   layerLoc->nBands = nf;

   return 0;
}

int HyPerCol::addLayer(HyPerLayer * l)
{
   assert(numLayers < maxLayers);
   l->columnWillAddLayer(icComm, numLayers);
   layers[numLayers++] = l;
   return (numLayers - 1);
}

int HyPerCol::addConnection(HyPerConn * conn)
{
   int connId = numConnections;

   assert(numConnections < maxConnections);

   // numConnections is the ID of this connection
   icComm->subscribe(conn);

   connections[numConnections++] = conn;

   return connId;
}

int HyPerCol::run(int nTimeSteps)
{
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
      icComm->deliver(this, layers[l]->getLayerId());

      // update layer and calculate new activity
      //
      layers[l]->updateState(sim_time, deltaTime);

      // TODO - move this to layer
      // Advance time level so we have a new place in data store
      // to copy the data.  This should be done immediately after before
      // publish so there is a place to publish and deliver the data to
      // but no one can access the data store (except to publish) until
      // wait has been called.  This should be fixed so that publish goes
      // to last time level and level is advanced only after wait.
      icComm->increaseTimeLevel(layers[l]->getLayerId());

      layers[l]->publish(icComm, sim_time);
   }

   // wait for all published data to arrive
   //
   for (int l = 0; l < numLayers; l++) {
      icComm->wait(layers[l]->getLayerId());
   }

   // make sure simTime is updated even if HyPerCol isn't running time loop
   simTime = sim_time + deltaTime;

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

int HyPerCol::initializeThreads()
{
   int err = 0;
#ifdef IBM_CELL_BE
   err = pv_cell_thread_init(columnId(), numThreads);
#endif
   return err;
}

int HyPerCol::finalizeThreads()
{
   int err = 0;
#ifdef IBM_CELL_BE
   err = pv_cell_thread_finalize(columnId(), numThreads);
#endif
   return err;
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

} // PV namespace
