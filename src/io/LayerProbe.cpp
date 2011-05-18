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
   char * outputdir = hc->getOutputPath();
   char * path = (char *) malloc(strlen(outputdir)+1+strlen(filename)+1);
   sprintf(path, "%s/%s", outputdir, filename);
   fp = fopen(path, "w");
   if( !fp ) {
      fprintf(stderr, "LayerProbe: Unable to open \"%s\" for writing.  Error %d\n", path, errno);
      exit(EXIT_FAILURE);
   }
   free(path);
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
