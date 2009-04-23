/*
 * PVLayerProbe.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "PVLayerProbe.hpp"

namespace PV {

PVLayerProbe::PVLayerProbe()
{
   fp = stdout;
}

PVLayerProbe::PVLayerProbe(const char * filename)
{
   char path[PV_PATH_MAX];
   sprintf(path, "%s%s", OUTPUT_PATH, filename);
   fp = fopen(path, "w");
}

PVLayerProbe::~PVLayerProbe()
{
   if (fp != NULL && fp != stdout) {
      fclose(fp);
   }
}

int PVLayerProbe::outputState(float time, PVLayer * l)
{
   return 0;
}

} // namespace PV
