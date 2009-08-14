/*
 * PostConnProbe.cpp
 *
 *  Created on: May 12, 2009
 *      Author: rasmussn
 */

#include "PostConnProbe.hpp"
#include <assert.h>

namespace PV {

/**
 * @kPost
 */
PostConnProbe::PostConnProbe(int kPost)
   : ConnectionProbe(0)
{
   this->kPost = kPost;
}

/**
 * @filename
 * @kPost
 */
PostConnProbe::PostConnProbe(const char * filename, int kPost)
   : ConnectionProbe(filename, 0)
{
   this->kPost = kPost;
}

/**
 * @time
 * @c
 */
int PostConnProbe::outputState(float time, HyPerConn * c)
{
   PVPatch ** wPost = c->convertPreSynapticWeights(time);
   PVPatch * w = wPost[kPost];

   fprintf(fp, "w%d:     ", kPost);
   fprintf(fp, "w=");
   text_write_patch(fp, w, w->data);
   fprintf(fp, "\n");
   fflush(fp);

   return 0;
}

} // namespace PV
