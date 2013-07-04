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
   initLayerProbe_base();
   // Derived classes of LayerProbe should call LayerProbe::initialize themselves.
}

/**
 * @filename
 */
LayerProbe::LayerProbe(const char * filename, HyPerLayer * layer)
{
   initLayerProbe_base();
   initLayerProbe(filename, layer);
}

LayerProbe::~LayerProbe()
{
   if (outputstream != NULL) {
      PV_fclose(outputstream);
   }
}

int LayerProbe::initLayerProbe_base() {
   outputstream = NULL;
   targetLayer = NULL;
   return PV_SUCCESS;
}

/**
 * @filename
 * @layer
 */
int LayerProbe::initLayerProbe(const char * filename, HyPerLayer * layer)
{
   setTargetLayer(layer);
   initOutputStream(filename, layer);
   layer->insertProbe(this);
   return PV_SUCCESS;
}

int LayerProbe::initOutputStream(const char * filename, HyPerLayer * layer) {
   HyPerCol * hc = layer->getParent();
   if( hc->columnId()==0 ) {
      if( filename != NULL ) {
         char * outputdir = hc->getOutputPath();
         char * path = (char *) malloc(strlen(outputdir)+1+strlen(filename)+1);
         sprintf(path, "%s/%s", outputdir, filename);
         bool append = layer->getParent()->getCheckpointReadFlag();
         const char * fopenstring = append ? "a" : "w";
         outputstream = PV_fopen(path, fopenstring);
         if( !outputstream ) {
            fprintf(stderr, "LayerProbe error opening \"%s\" for writing: %s\n", path, strerror(errno));
            exit(EXIT_FAILURE);
         }
         free(path);
      }
      else {
         outputstream = PV_stdout();
      }
   }
   else {
      outputstream = NULL; // Only root process writes; if other processes need something written it should be sent to root.
                           // Derived classes for which it makes sense for a different process to do the file i/o should override initOutputStream
   }
   return PV_SUCCESS;
}

/**
 * @time
 */
int LayerProbe::outputState(double timef)
{
   return 0;
}

} // namespace PV
