/*
 * pv_thread.h
 *
 *  Created on: Dec 6, 2008
 *      Author: rasmussn
 */

#ifndef PV_THREAD_H_
#define PV_THREAD_H_

#include "../../include/pv_arch.h"

#ifdef PV_USE_PTHREADS

#include "../../include/pv_types.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>  /* or <inttypes.h> (<sys/types.h> doesn't work on Darwin) */

#define ADDR_MASK       0xfffffff0
#define CODE_MASK       0xf

#ifdef IBM_CELL_BE
#  include <libspe2.h>
#  define pv_thread_context_ptr_t spe_context_ptr_t
#else
#  define pv_thread_context_ptr_t int
#endif

typedef struct {
   int tid;
   int column;          // column id
   pthread_t pthread;
   pv_thread_context_ptr_t ctx;
   void * tinfo;
} pv_thread_env_t;

/* TODO - tinfo above should be of this type? */
/* TODO - make sure this is properly aligned */
typedef struct {
   int id;
   size_t max_recv_size;
   float * data;
   int padding;
} pv_thread_params_t;

/* global parameters (WARNING - accessed from multiple threads) */

extern pv_thread_env_t thread_env_g[MAX_THREADS];  /* used by main or main_ppu only */


#ifdef CHARLES
typedef struct {
   uint32_t myspu;
   uint32_t num_steps;
   uint32_t xpe, ype;
   uint32_t mype, numpe;
   int32_t nbr_list[9];
   uint32_t dummy;  /* for alignment */
   uint64_t ppu_phdata_ea;
   uint64_t ppu_xdata_ea;
   uint64_t ppu_ydata_ea;
   uint64_t ppu_thdata_ea;
   uint64_t ppu_xpdata_ea;
   uint64_t ppu_ypdata_ea;
   uint64_t ppu_fdata_ea;
   uint64_t ppu_timing_ea;
   uint64_t ppu_mboard_ea;
   uint64_t prev_ls_base_ea;
   uint64_t next_ls_base_ea;
   uint64_t dummy64;  /* for alignment */
} spu_param_t;

typedef struct{
   uint64_t cyc_compute;
   uint64_t cyc_p2s_get;
   uint64_t cyc_s2s_put;
   uint64_t cyc_s2p_put;
} spu_timing_t;

typedef struct{
   float ph[CHUNK_SIZE];
   float x[CHUNK_SIZE];
   float y[CHUNK_SIZE];
   float dummy[CHUNK_SIZE]; /* for alignment */
} data_chunk_t;
#endif /* CHARLES */

#ifdef __cplusplus
extern "C" {
#endif

int pv_thread_init(int column, int numThreads);
int pv_thread_finalize(int column, int numThreads);

int pv_signal_threads_recv(void * addr, unsigned char msg);

int pv_in_mbox_write(pv_thread_context_ptr_t ctx,
                     unsigned int * mbox_data, int count, unsigned int behavior);

static inline
uint32_t encode_msg(void * buf, char code)
{
#ifdef PV_ARCH_64
   /* send only the low portion of the address */
   assert( ((uint64_t) buf & ADDR_MASK) == ((uint64_t) buf & 0xffffffff) );
   return (((uint64_t) buf) & 0xffffffff) | (code & CODE_MASK);
#else
   assert(((uint32_t) buf & ADDR_MASK) == (uint32_t) buf);
   return ((uint32_t) buf) | (code & CODE_MASK);
#endif

}

static inline
char decode_msg(uint32_t msg, void ** addr)
{
#ifdef PV_ARCH_64
   *addr = (void*) ((uint64_t) msg & ADDR_MASK);
#else
   *addr = (void*) (msg & ADDR_MASK);
#endif
   return  (msg & CODE_MASK);
}

#ifdef __cplusplus
}
#endif

#endif /* PV_USE_PTHREADS */

#endif /* PV_THREAD_H_ */
