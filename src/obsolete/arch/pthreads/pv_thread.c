/*
 * pv_thread.c
 *
 *  Created on: Dec 6, 2008
 *      Author: rasmussn
 */

#include "pv_thread.h"

#ifdef PV_USE_PTHREADS

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* mailbox codes */
#define PV_MSG_START  (-1 & CODE_MASK)
#define PV_MSG_EMPTY  (-2 & CODE_MASK)
#define PV_MSG_EXIT   (-3 & CODE_MASK)

#undef DEBUG_THREADS
#undef DEBUG_THREAD_YIELD

/* global parameters (WARNING - accessed from multiple threads) */

// TODO - alignment
pv_thread_env_t    thread_env_g[MAX_THREADS];      /* used by main or main_ppu only */
pv_thread_params_t thread_params_g[MAX_THREADS];   /* a separate one for each thread */

int numThreads_g;
volatile int in_mbox_g[MAX_THREADS] __attribute__ ((aligned(128)));

pthread_attr_t  attr_g;
pthread_mutex_t mbox_mutex_g;
pthread_cond_t  mbox_cv_g;

void * pv_pthread_run(void * tinfo);
int thread_env_init(pv_thread_env_t * env, pv_thread_params_t * params, int numThreads, int column);
int thread_params_init(pv_thread_env_t * env, int numThreads);

/**
 * @addr
 * @size
 */

static inline
void * get_data(void * addr, size_t size)
{
   return addr;
}
/**
 * @column
 * @numThreads
 */

int pv_thread_init(int column, int numThreads)
{
   int t;
//   void *(*fn)(void *);

   numThreads_g = numThreads;

   /* Initialize mutex and condition variable objects */
   pthread_mutex_init(&mbox_mutex_g, NULL);
   pthread_cond_init (&mbox_cv_g, NULL);

   /* For portability, explicitly create threads in a joinable state */
   pthread_attr_init(&attr_g);
   pthread_attr_setdetachstate(&attr_g, PTHREAD_CREATE_JOINABLE);

   // initialize mailboxes
   for (t = 0; t < MAX_THREADS; t++) {
      in_mbox_g[t] = PV_MSG_START;
   }

   /* initialize thread environment */
   thread_env_init(thread_env_g, thread_params_g, numThreads, column);

   /* initialize thread parameters */
   thread_params_init(thread_env_g, numThreads);

   /* create pthreads */
   for (t = 0; t < numThreads; t++) {
      thread_env_g[t].tid = t;
      thread_env_g[t].column = column;
      if (pthread_create(&thread_env_g[t].pthread, &attr_g, &pv_pthread_run, &thread_env_g[t])) {
         perror("Failed creating thread");
         exit(1);
      }
   }

   /* wait for threads to check in */
   for (t = 0; t < numThreads; t++) {
      while (in_mbox_g[t] != PV_MSG_EMPTY) {
#ifdef DEBUG_THREAD_YIELD
         printf("pthread_init: waiting for threads to checkin, mbox=%d\n", in_mbox_g[t]);
#endif
         sched_yield();
      }
   }

   return 0;
}

/**
 * @column
 * @numThreads
 */

int pv_thread_finalize(int column, int numThreads)
{
   int t;
   int err = 0;

   // signal threads to exit
   err = pv_signal_threads_recv(NULL, PV_MSG_EXIT);

   pthread_attr_destroy(&attr_g);
   pthread_mutex_destroy(&mbox_mutex_g);
   pthread_cond_destroy(&mbox_cv_g);

   for (t = 0; t < numThreads; t++) {
      if (pthread_join(thread_env_g[t].pthread, NULL)) {
         perror("pv_thread_finalize:FAILED joining thread");
         err = -1;
      }
   }

   pthread_exit(NULL);

   return err;
}

/**
 * @params
 * @msg
 */

static
int handle_run_msg(pv_thread_params_t * params, uint32_t msg)
{
   void * addr;
   char code = decode_msg(msg, &addr);


   if (code == PV_MSG_EXIT) {
      printf("handle_run_msg: exiting: msg=%d, addr=%p\n", code, addr);
      return 0;
   }

   /* only recv for now */

   PVLayerCube * activity = (PVLayerCube *) get_data(addr, params->max_recv_size);
   printf("handle_run_msg: activity.size=%ld\n", activity->size);

   return 0;
}

/**
 * @tinfo
 */

void * pv_pthread_run(void * tinfo)
{
   int msg;
   pv_thread_env_t * env = (pv_thread_env_t *) tinfo;
   int tid = env->tid;

   msg = in_mbox_g[tid];
#ifdef DEBUG_THREADS
   printf("[%d]: pv_pthread_run: running thread %d, initial msg=%p\n", env->column, tid, (void *) msg);
#endif

   /* notify parent that thread is running */
   pthread_mutex_lock(&mbox_mutex_g);
   in_mbox_g[tid] = PV_MSG_EMPTY;
   pthread_mutex_unlock(&mbox_mutex_g);

   while (msg != PV_MSG_EXIT) {
      /* wait for message to arrive */
      while (in_mbox_g[tid] == PV_MSG_EMPTY) {
#ifdef DEBUG_THREAD_YIELD
         printf("[%d]: pv_pthread_run: waiting for msg, current=%p\n",
                env->column, (void *) in_mbox_g[tid]);
#endif
         sched_yield();
      }

      pthread_mutex_lock(&mbox_mutex_g);

      //pthread_cond_wait(&mbox_cv_g, &mbox_mutex_g);
      msg = in_mbox_g[tid];
      in_mbox_g[tid] = PV_MSG_EMPTY;       // reset mailbox
      pthread_mutex_unlock(&mbox_mutex_g);

#ifdef DEBUG_THREADS
      printf("[%d]: pv_pthread_run::::: thread %d received msg %p\n", env->column, tid, (void *) msg); fflush(stdin);
#endif

      handle_run_msg(tinfo, msg);
   }

   fprintf(stderr, "[%d]: pv_pthread_run: exiting thread %d\n", env->column, tid);

   pthread_exit(NULL);
}

/**
 * @addr
 * @msg
 */
int pv_signal_threads_recv(void * addr, unsigned char msg)
{
   int t;
   int err = 0;

   uint32_t encoded_msg = encode_msg(addr, msg);

   for (t = 0; t < numThreads_g; t++) {
      err = pv_in_mbox_write(thread_env_g[t].ctx, &encoded_msg, 1, 1);
   }

   return err;
}

/**
 * @ctx
 * @mbox_data
 * @count
 * @behavior
 */
int pv_in_mbox_write(pv_thread_context_ptr_t ctx,
                     unsigned int * mbox_data, int count, unsigned int behavior)
{
   // TODO - fix to make use of count?
   assert(count == 1);

   /* wait for mbox to come open, just reading no need to lock */
   while (in_mbox_g[ctx] != PV_MSG_EMPTY) {
#ifdef DEBUG_THREAD_YIELD
      printf("pv_in_mbox_write: waiting for mail box to come free, msg[%d] = %p\n",
             ctx, (void *) in_mbox_g[ctx]);
#endif
      sched_yield();
   }

   pthread_mutex_lock(&mbox_mutex_g);
#ifdef DEBUG_THREADS
   printf("pv_in_mbox_write: signaling thread %d, msg=%p.\n", ctx, (void *) mbox_data[0]); fflush(stdin);
#endif

   in_mbox_g[ctx] = mbox_data[0];
   //pthread_cond_signal(&mbox_cv_g);
   pthread_mutex_unlock(&mbox_mutex_g);

   return 0;
}

/**
 * @env
 * @params
 * @numThreads
 * @column
 */
int thread_env_init(pv_thread_env_t * env,
                    pv_thread_params_t * params, int numThreads, int column)
{
   int t;
   for (t = 0; t < numThreads; t++) {
      env[t].tid = t;
      env[t].ctx = t;          /* for pthreads only */
      env[t].column = column;
      env[t].pthread = 0;      /* initialized later */
      env[t].tinfo = &params[t];
   }

   return 0;
}

/**
 * @env
 * @numThreads
 */
int thread_params_init(pv_thread_env_t * env, int numThreads)
{
   int t;
   for (t = 0; t < numThreads; t++) {
      pv_thread_params_t * tinfo = env[t].tinfo;
      tinfo->id = env[t].tid;
      tinfo->max_recv_size = 64*64*8*4;  // TODO - FIXME
   }

   return 0;
}

#else

void pv_thread_noop() { }

#endif /* PV_USE_PTHREADS */
