/*
 * HyPerCol.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: rasmussn
 */

#include "HyPerCol.hpp"
#include "InterColComm.hpp"
#include "../arch/pthreads/pv_thread.h"
#include "../connections/PVConnection.h"
#include "../connections/WeightCache.h"
#include "../io/io.h"

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>

namespace PV {

HyPerCol::HyPerCol(const char * name, int argc, char * argv[])
{
   // TODO - fix these numbers to dynamically grow
   maxLayers = MAX_LAYERS;
   maxConnections = MAX_CONNECTIONS;

   this->name = strdup(name);

   char * param_file;
   currTime = 0;
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

   icComm = new InterColComm(&argc, &argv, this);

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
   int stop = currTime + nTimeSteps;
   numSteps = nTimeSteps;

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol: running...\n");  fflush(stdout);
   }
#endif

   // publish initial conditions
   for (int l = 0; l < numLayers; l++) {
      layers[l]->publish(icComm, (float) currTime);
   }

   while (currTime++ < stop) {

      // if (columnId() == 0) {
      //    printf("\n[0]: HyPerCol::run: beginning timestep %d\n", currTime);  fflush(stdout);
      // }

      // deliver published data for each layer
      for (int l = 0; l < numLayers; l++) {
         // this function blocks until all data for a layer has been delivered
#ifdef DEBUG_OUTPUT
         printf("[%d]: HyPerCol::run will deliver layer %d\n", columnId(), l);
         fflush(stdout);
#endif
         icComm->deliver(l);
      }

#ifdef DEBUG_OUTPUT
      if (columnId() == 0) {
         printf("[0]: HyPerCol::run: data delivery finished\n");  fflush(stdout);
      }
#endif

      for (int l = 0; l < numLayers; l++) {
         layers[l]->updateState((float) currTime, deltaTime);
         icComm->increaseTimeLevel(layers[l]->getLayerId());
         layers[l]->publish(icComm, (float) currTime);
      }

   }  // end run loop

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol::run done...\n");  fflush(stdout);
   }
#endif

   return 0;
}

int HyPerCol::run_old(int nTimeSteps)
{
   int stop = currTime + nTimeSteps;
   run_struct * tinfo;

#ifdef MULTITHREAD
   pthread_t * thread;
   static pthread_attr_t pattr;
#endif

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]:HyPerCol: running...\n");  fflush(stdout);
   }
#endif

   int numTInfo = (numConnections > numLayers) ? numConnections : numLayers;
   tinfo = (run_struct *) malloc(sizeof(run_struct) * numThreads * numTInfo);

#ifdef MULTITHREAD
   thread = (pthread_t *) malloc(sizeof(pthread_t) * numThreads * numTInfo);
   pthread_attr_init( &pattr );
   pthread_attr_setdetachstate( &pattr, PTHREAD_CREATE_JOINABLE );
#endif

   while (currTime++ < stop) {

#ifdef DEBUG_OUTPUT
      if (columnId() == 0) {
         printf("\n[0]:HyPerCol::run: beginning timestep %d\n", currTime);  fflush(stdout);
      }
#endif

      // each layer should deliver it data (previously published)
      for (int ic = 0; ic < numLayers; ic++) {
         icComm->deliver(ic);
      }

      // TODO - WARNNG - make sure that this works with initializeThreads (using numConnections)

      // For each connection, pass the activity
      for (int c = 0; c < numConnections; c++) {
         if (connections[c]->pvconn->recvFunc == NULL) {
//            HyPerConn *  conn = connections[c];
//            HyPerLayer * pre  = conn->pre;
//            conn->post->recvSynapticInput(pre, pre->clayer->numNeurons,
//                                          pre->clayer->fActivity[conn->pvconn->readIdx]);
//            conn->post->recvSynapticInput(pre, pre->clayer->activity);
         }
         else {
            // we have a recvFunc, use it instead of C++ method
            for (int t = 0; t < numThreads; t++) {
               int ith = c * numThreads + t;
               tinfo[ith].hc = this;
               tinfo[ith].layer = connections[c]->pvconn->post->layerId * numThreads + t;
               tinfo[ith].proc = c;

#ifdef OLD_MULTITHREADED
               if (pthread_create(&thread[ith], &pattr, run1connection, (void*)&tinfo[ith] ))
               {
                  // TODO - fail gracefully with MPI
                  fprintf(stderr, "[%d]: pthread %d failed\n", columnId(), c);
                  exit(-1);
               }
#else
               run1connection(&tinfo[ith]);
#endif
            } // end thread creation loop

#ifdef OLD_MULTITHREADED
            // must wait for all threads of this connection to complete
            for (int t = 0; t < numThreads; t++) {
               void * retval;
               int ith = c * numThreads + t;
               if (pthread_join(thread[ith], &retval))
               {
                  // TODO -fail gracefully with MPI
                  fprintf(stderr, "[%d]: pthread join %d failed\n", columnId(), ith);
                  exit(-1);
               }
            } // end thread join loop
#endif
         } // end forall connections loop
      } // end while (currTime < stop) loop

      // Now, all synaptic input has been received.
#ifdef DEBUG_OUTPUT
      if (columnId() == 0) {
         printf("[0]:HyPerCol::run: connections finished\n");  fflush(stdout);
      }
#endif

      // update each layer
      for (int l = 0; l < this->numLayers; l++) {
         PVLayer * clayer = layers[l]->getCLayer();

         if (clayer->updateFunc) {
            for (int t = 0; t < numThreads; t++) {
               int ith = l * numThreads + t;
               tinfo[ith].hc = this;
               tinfo[ith].layer = l;
               tinfo[ith].proc = ith;

#ifdef OLD_MULTITHREADED
               if (pthread_create(&thread[ith], &pattr, update1layer, (void*)&tinfo[ith])) {
                  // TODO - fail gracefully with MPI
                  fprintf(stderr, "[%d]: pthread U %d failed\n", columnId(), ith);
                  exit(-1);
               }
#else
               update1layer((void*) &tinfo[ith]);
#endif
            } // end thread create

#ifdef OLD_MULTITHREADED
            // Don't really need to wait here, but it's easier implementation-wise.
            for (int t = 0; t < numThreads; t++) {
               void *retval;
               int ith = l * numThreads + t;
               if (pthread_join(thread[ith], &retval))
               {
                  // TODO - fail gracefully with MPI
                  fprintf(stderr, "[%d]: pthread join %d failed\n", columnId(), ith);
                  exit(-1);
               }
            }
#endif
         }
         else if (layers[l]) {
            // no clayer->updateFunc use virtual method
            // TODO - make this the default, but able to override with function pointers
            // C++ handler
            /* complete time step (i.e., update f & V) */
            layers[l]->updateState((float) currTime, deltaTime);
         } // For non-C, non parallel layers

         pvlayer_outputState(clayer);

         // TODO: use clayer->activeIndices
         //icComm->publish(clayer, clayer->numNeurons, clayer->fActivity[clayer->writeIdx]);
      } // for each layer

      // Do any ring buffer updates
      for (int c = 0; c < numConnections; c++) {
         connections[c]->pvconn->readIdx++;
         connections[c]->pvconn->readIdx %= MAX_F_DELAY;
      }

// TODO - WARNING - this looks wrong in terms of numConns vs numLayers and numThreads count
      for (int l = 0; l < this->numLayers; l++) {
         PVLayer * clayer = layers[l]->getCLayer();
         clayer->writeIdx = (clayer->writeIdx + 1) % clayer->numDelayLevels;
         for (int t = 0; t < numThreads; t++) {
            int ith = l * numThreads + t;
            threadCLayers[ith].writeIdx = (threadCLayers[ith].writeIdx + 1)
                  % threadCLayers[ith].numDelayLevels;
         }
      }

      for (int ic = 0; ic < this->numLayers; ic++) {
         icComm->deliver(ic);
      }

   }

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]:HyPerCol: done...\n");  fflush(stdout);
   }
#endif

   free(tinfo);

#ifdef OLD_MULTITHREADED
   free(thread);
   pthread_attr_destroy(&pattr);
#endif
   return 0;
}

int HyPerCol::initializeThreads()
{
   int err = 0;

#ifdef IBM_CELL_BE
   err = pv_cell_thread_init(columnId(), numThreads);
#else
   err = pv_thread_init(columnId(), numThreads);
#endif

#ifdef OLD
   // To parallelize on a shared memory multicore, create
   // copies of the layers for this single process to execute using pthreads.

   threadCLayers = (PVLayer *) malloc(sizeof(PVLayer) * numThreads * numLayers);

   for (int th = 0; th < numThreads; th++) {
      for (int l = 0; l < numLayers; l++) {
         p = &threadCLayers[l * numThreads + th];
         memcpy(p, layers[l]->getCLayer(), sizeof(PVLayer));

         // Exact copies (of buffers, handlers, etc.) *except* for the following:
         // TODO - check to see that divides properly for odd threads and ny
         p->loc.ny     /= numThreads;
         p->numNeurons /= numThreads;
         p->xOrigin = 0;
         p->yOrigin = th * p->loc.ny;
      }
#if 0
      // TODO - check to see if numLayers must be greater than numConnections
      for (l = 0; l < numConnections; l++)
      {
         c = &shmemConnections[l * procs + t];
         memcpy(c, connections[l], sizeof(PVConnection));
         c->post = &threadCLayers[l*procs+t];
      }
#endif //0
   }
#endif // OLD

   return err;
}

int HyPerCol::finalizeThreads()
{
   int err = 0;

#ifdef IBM_CELL_BE
   err = pv_cell_thread_finalize(columnId(), numThreads);
#else
   err = pv_thread_finalize(columnId(), numThreads);
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
      layers[l]->writeState(OUTPUT_PATH, currTime);
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

