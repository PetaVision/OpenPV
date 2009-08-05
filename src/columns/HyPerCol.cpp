/*
 * HyPerCol.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: rasmussn
 */

#undef TIMER_ON

#include "HyPerCol.hpp"
#include "InterColComm.hpp"
#include "../connections/PVConnection.h"
#include "../io/io.h"
#include "../io/clock.h"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

namespace PV {

HyPerCol::HyPerCol(const char * name, int argc, char * argv[])
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
   input_file = NULL;
   param_file = NULL;
   parse_options(argc, argv, &input_file, &param_file, &numSteps, &numThreads);

   // estimate for now
   // TODO -get rid of maxGroups
   int maxGroups = 2*(maxLayers + maxConnections);
   params = new PVParams(param_file, maxGroups);

   icComm = new InterColComm(&argc, &argv);

   if (param_file != NULL) free(param_file);

   deltaTime = DELTA_T;
   if (params->present(name, "dt")) deltaTime = params->value(name, "dt");

   imageRect.x = 0.0;
   imageRect.y = 0.0;
   imageRect.width  = params->value(name, "nx");
   imageRect.height = params->value(name, "ny");
}

HyPerCol::~HyPerCol()
{
   int n;

#ifdef MULTITHREADED
   finalizeThreads();
#endif

   if (input_file != NULL) free(input_file);

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
   int err = 0;

   for (int i = 0; i < this->numLayers; i++) {
      err = layers[i]->initFinish();
      if (err != 0) {
         fprintf(stderr, "[%d]: HyPerCol::initFinish: ERROR condition, exiting...\n", this->columnId());
         exit(err);
      }
   }

   log_parameters(numSteps, input_file);

#ifdef MULTITHREADED
   initializeThreads();
#endif

   return err;
}

int HyPerCol::columnId()
{
   return icComm->rank();
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
   int stop = nTimeSteps;
   numSteps = nTimeSteps;

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol: running...\n");  fflush(stdout);
   }
#endif

   // publish initial conditions
   for (int l = 0; l < numLayers; l++) {
      layers[l]->publish(icComm, time);
   }

   while (step++ < stop) {

#ifdef TIMER_ON
      if (step == 10) start_clock();
#endif

      // deliver published data for each layer
      for (int l = 0; l < numLayers; l++) {
         // this function blocks until all data for a layer has been delivered
#ifdef DEBUG_OUTPUT
         printf("[%d]: HyPerCol::run will deliver layer %d\n", columnId(), l);
         fflush(stdout);
#endif
         icComm->deliver(this, l);
      }

#ifdef DEBUG_OUTPUT
      if (columnId() == 0) {
         printf("[0]: HyPerCol::run: data delivery finished\n");  fflush(stdout);
      }
#endif

      for (int l = 0; l < numLayers; l++) {
         layers[l]->updateState(time, deltaTime);
         icComm->increaseTimeLevel(layers[l]->getLayerId());
         layers[l]->publish(icComm, time);
      }

      // layer activity has been calculated, inform connections
      for (int c = 0; c < numConnections; c++) {
         connections[c]->updateState(time, deltaTime);
         connections[c]->outputState(time);
      }

      time += deltaTime;
   }  // end run loop

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol::run done...\n");  fflush(stdout);
   }
#endif

#ifdef TIMER_ON
      stop_clock();
#endif

   return 0;
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

}; // PV namespace

extern "C" {

void * run1connection(void * arg)
{
#ifdef DELETE
   int layers = ((run_struct *) arg)->layers;
   int proc = ((run_struct *) arg)->proc;
   PV::HyPerCol * hc = ((run_struct *) arg)->hc;
   clock_t ticks;
   ticks = clock();
   srand(ticks + proc);

   printf("c%d start: %ld\n", i, ticks);
   hc->connections[layers].recvFunc(hc->connections[layers].pre,
   	hc->connections[proc].recvFunc(&hc->connections[proc],
   	&hc->shmemCLayers[layers], hc->connections[proc].pre->numNeurons,
   	hc->connections[proc].pre->fActivity[hc->connections[proc].readIdx]);
   ticks = clock() - ticks;
   printf("c%d diff : %ld\n", i, ticks);
#ifdef MULTITHREADED
   pthread_exit(0);
#endif
#endif
   return NULL;
}

void * update1layer(void * arg)
{
#ifdef DELETE
   run_struct * info = (run_struct *) arg;

   int itl = info->proc;
   PV::HyPerCol * hc = info->hc;

   clock_t ticks;
   ticks = clock();
   srand(ticks + itl);
   hc->threadCLayers[itl].updateFunc(&hc->threadCLayers[itl]);
#ifdef MULTITHREADED
   pthread_exit(0);
#endif
#endif
   return NULL;
}

}
; // extern "C"

