/*
 * CocircConn.cpp
 *
 *  Created on: Nov 10, 2008
 *      Author: rasmussn
 */

#ifdef OBSOLETE // Marked obsolete Aug 4.  Use KernelConn with cocircWeights set to true in the params file

#include "CocircConn.hpp"
#include "../io/io.h"
#include "../utils/conversions.h"
#include <assert.h>
#include <string.h>

namespace PV {

CocircConn::CocircConn()
{
   printf("CocircConn::CocircConn: running default constructor\n");
   initialize_base();
}

CocircConn::CocircConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL);
}

CocircConn::CocircConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL); // use default channel
}

// provide filename or set to NULL
CocircConn::CocircConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename);
}

PVPatch ** CocircConn::initializeDefaultWeights(PVPatch ** patches, int numPatches)
{
   return initializeCocircWeights(patches, numPatches);
}

PVPatch ** CocircConn::initializeCocircWeights(PVPatch ** patches, int numPatches)
{
   return HyPerConn::initializeCocircWeights(patches, numPatches);
}

int CocircConn::cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
      float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
      float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
      float aspect, float rotate, float sigma, float r2Max, float strength)
{
   return HyPerConn::cocircCalcWeights(wp, kPre, noPre, noPost,
         sigma_cocirc, sigma_kurve, sigma_chord, delta_theta_max,
         cocirc_self, delta_radius_curvature, numFlanks, shift,
         aspect, rotate, sigma, r2Max, strength);
}

} // namespace PV

#endif // OBSOLETE
