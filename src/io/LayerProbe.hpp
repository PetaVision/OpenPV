/*
 * LayerProbe.h
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#ifndef LAYERPROBE_HPP_
#define LAYERPROBE_HPP_

#include <stdio.h>

namespace PV {

class HyPerCol;
class HyPerLayer;

typedef enum {
   BufV,
   BufActivity
} PVBufType;

class LayerProbe {

// Methods
public:
   LayerProbe(const char * filename, HyPerLayer * layer);
   virtual ~LayerProbe();

   virtual int outputState(float timef) = 0;

   HyPerLayer * getTargetLayer() {return targetLayer;}

protected:
   LayerProbe();
   int initLayerProbe(const char * filename, HyPerLayer * layer);
   virtual int initFilePointer(const char * filename, HyPerLayer * layer);

private:
   int initialize_base();
   void setTargetLayer(HyPerLayer * l) {targetLayer = l;}

// Member variables
protected:
   FILE * fp;

private:
   HyPerLayer * targetLayer;
};

}

#endif /* LAYERPROBE_HPP_ */
