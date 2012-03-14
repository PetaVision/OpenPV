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
   // Derived classes of LayerProbe should call LayerProbe::initialize themselves.
}

/**
 * @filename
 */
LayerProbe::LayerProbe(const char * filename, HyPerLayer * layer)
{
   initLayerProbe(filename, layer);
}

LayerProbe::~LayerProbe()
{
   if (fp != NULL && fp != stdout) {
      fclose(fp);
   }
}

/**
 * @fp
 * @l
 */
int LayerProbe::initLayerProbe(const char * filename, HyPerLayer * layer)
{
   HyPerCol * hc = layer->getParent();
   if( hc->icCommunicator()->commRank()==0 ) {
      if( filename != NULL ) {
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
      else {
         fp = stdout;
      }
   }
   else {
      fp = NULL; // Only root process should be writing; if other processes need something written it should be sent to root.
   }
   setTargetLayer(layer);
   layer->insertProbe(this);
   return PV_SUCCESS;
}

/**
 * @time
 */
int LayerProbe::outputState(float timef)
{
   return 0;
}

} // namespace PV
