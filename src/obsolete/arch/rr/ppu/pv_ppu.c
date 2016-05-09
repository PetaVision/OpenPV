/*
 * pv_ppu.c
 *
 *  Created on: Dec 5, 2008
 *      Author: rasmussn
 */

#include "../../include/pv_arch.h"
#include "../../pthreads/pv_thread.h"

extern spe_program_handle_t pv_spu;

/**
 * @arg
 */
void * ppu_pthread_run(void * arg)
{
   thread_env_t * env = (thread_env_t *) arg;
   unsigned int entry = SPE_DEFAULT_ENTRY;
   if (spe_context_run(env->ctx, &entry, 0, env->tinfo, NULL, NULL) < 0) {
      perror ("Failed running context");
      exit (1);
   }

   pthread_exit(NULL);
}

/**
 * @numThreads
 */
int pv_cell_thread_init(int numThreads)
{
   int spu;
   int err = 0;

   assert(numThreads <= MAX_THREADS);

   if (spe_cpu_info_get(SPE_COUNT_PHYSICAL_SPES, -1) < numThreads) {
     fprintf(stderr, "pv_cell_thread_init: number SPEs < numThreads=%d\n", numThreads);
     return -1;
   }

   for (spu = 0; spu < numThreads; spu++) {
      /* create the SPE context */
      if ((thread_env_g[spu].ctx = spe_context_create(0, NULL)) == NULL) {
         perror("pv_cell_thread_init: FAILED to create SPE context");
         return -1;
      }

      /* load the program into the context */
      if (spe_program_load(thread_env_g[spu].ctx, &pv_spu) != 0) {
        perror("pv_cell_thread_init: FAILED to load program");
        return -2;
      }
   }

   /* map memory */

   /* set the thread environment */

   /* create pthreads */
   for (spu = 0; spu < numThreads; spu++) {
      if (pthread_create(&thread_env_g[spu].pthread, NULL, &ppu_pthread_run, &thread_env_g[spu])) {
         perror("Failed creating thread");
         exit(1);
      }
   }

   /* ready to run ppu event loop */

   return err;
}

/**
 * @numThreads
 */
int pv_cell_thread_finalize(int numThreads)
{
   int spu;
   int err = 0;

   for (spu = 0; spu < numThreads; spu++) {
      if (pthread_join(thread_env_g[spu].pthread, NULL)) {
         perror("pv_cell_thread_finalized:FAILED joining thread");
         exit(1);
      }

      if (spe_context_destroy(thread_env_g[spu].ctx)) {
         perror("pv_cell_thread_finalize: FAILED to destroy context");
         exit(1);
      }
   }

#ifdef TUTORIAL
   /* check the SPE status */
   if (stop_info.stop_reason == SPE_EXIT) {
     if (stop_info.result.spe_exit_code != 0) {
       fprintf(stderr, "FAILED: SPE returned a non-zero exit status\n");
       exit(1);
     }
   } else {
     fprintf(stderr, "FAILED: SPE abnormally terminated\n");
     exit(1);
   }
#endif
}
