/*
 * LayerProbe.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "LayerProbe.hpp"
#include "../layers/HyPerLayer.hpp"

namespace PV {

LayerProbe::LayerProbe()
{
   fp = stdout;
}

/**
 * @filename
 */
LayerProbe::LayerProbe(const char * filename, HyPerCol * hc)
{
   char path[PV_PATH_MAX];
   sprintf(path, "%s/%s", hc->getOutputPath(), filename);
   fp = fopen(path, "w");
}

LayerProbe::~LayerProbe()
{
   if (fp != NULL && fp != stdout) {
      fclose(fp);
   }
}

/**
 * @time
 * @l
 */
int LayerProbe::outputState(float time, HyPerLayer * l)
{
   return 0;
}

} // namespace PV
