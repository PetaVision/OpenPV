/*
 * pv_cell.h
 *
 *  Created on: Dec 6, 2008
 *      Author: rasmussn
 */

#ifndef PV_CELL_H_
#define PV_CELL_H_

typedef struct {
   spe_context_ptr_t spe_ctx;
   pthread_t pthread;
   void * argp;
} pv_thread_env_t;

#endif /* PV_CELL_H_ */
