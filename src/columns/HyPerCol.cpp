/*
 * HyPerCol.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#undef TIMER_ON

#include "HyPerCol.hpp"
#include "InterColComm.hpp"
#include "../connections/PVConnection.h"
#include "../io/clock.h"
#include "../io/imageio.hpp"
#include "../io/io.h"

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
   time = 0;
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
   float stopTime = time + nTimeSteps * deltaTime;
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
      layers[l]->publish(icComm, time);
   }

   if (runDelegate) {
      // let delegate advance the time
      //
      runDelegate->run(time, stopTime);
   }

   // time loop
   //
   while (time < stopTime) {
      time = advanceTime(time);
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

float HyPerCol::advanceTime(float simTime)
{
   // deliver published data for each layer
   //
   for (int l = 0; l < numLayers; l++) {
#ifdef DEBUG_OUTPUT
      printf("[%d]: HyPerCol::run will deliver layer %d\n", columnId(), l);
      fflush(stdout);
#endif
      // this function blocks until all data for a layer has been delivered
      //
      icComm->deliver(this, l);
   }

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol::run: data delivery finished\n");  fflush(stdout);
   }
#endif

   for (int l = 0; l < numLayers; l++) {
      layers[l]->updateState(simTime, deltaTime);
      layers[l]->outputState(simTime+deltaTime);
      icComm->increaseTimeLevel(layers[l]->getLayerId());
      layers[l]->publish(icComm, simTime);
   }

   // layer activity has been calculated, inform connections
   for (int c = 0; c < numConnections; c++) {
      connections[c]->updateState(simTime, deltaTime);
      connections[c]->outputState(simTime+deltaTime);
   }

   return simTime + deltaTime;
}

int HyPerCol::exitRunLoop(bool exitOnFinish)
{
   int status = 0;

   // output final state of layers and connections
   //
   bool last = true;

   for (int l = 0; l < numLayers; l++) {
      layers[l]->writeState(layers[l]->getName(), time, last);
   }

   for (int c = 0; c < numConnections; c++) {
      connections[c]->outputState(time, last);
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
      layers[l]->writeState(OUTPUT_PATH, time);
   }
   return 0;
}

} // PV namespace
