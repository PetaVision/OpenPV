/*
 * PVConnection.c
 *
 *  Created on: Jul 29, 2008
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // for memcpy
#include "../include/pv_common.h"
#include "PVConnection.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef USE_PVCONNECTION

// TODO - use HyPerConn class to do much of this?
int pvConnInit(PVConnection * pvConn, PVLayer * pre, PVLayer * post, PVConnParams * p, int channel)
{
   pvConn->pre  = pre;
   pvConn->post = post;

   pvConn->whichPhi = channel;

   // if this is not null, there is a free error (not used anyway for now)
   pvConn->params   = NULL;

   pvConn->delay       = p->delay;
   pvConn->fixDelay    = p->fixDelay;
   pvConn->varDelayMin = p->varDelayMin;
   pvConn->varDelayMax = p->varDelayMax;
   pvConn->numDelay    = p->numDelay;
   pvConn->isGraded    = p->isGraded;
   pvConn->vel         = p->vel;
   pvConn->rmin        = p->rmin;
   pvConn->rmax        = p->rmax;

   pvConn->preNormF  = NULL;
   pvConn->postNormF = NULL;

   // make the cutoff boundary big for now
   pvConn->r2 = post->loc.nx * post->loc.ny + post->loc.ny * post->loc.ny;

   // TODO - use numDelayLevels rather than MAX_F_DELAY
   // Init the read ptr such that it follows the writeIdx (which
   // start at 0) by the correct amount.
   pvConn->readIdx = (MAX_F_DELAY - pvConn->delay - 1);

   return 0;
}

int pvConnFinalize(PVConnection * pvConn)
{
   if (pvConn->params)    free(pvConn->params);
   if (pvConn->preNormF)  free(pvConn->preNormF);
   if (pvConn->postNormF) free(pvConn->postNormF);

   return 0;
}
#endif // USE_PVCONNECTION

#ifdef __cplusplus
}
#endif
