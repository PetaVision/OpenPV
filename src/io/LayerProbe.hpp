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
public:
   LayerProbe();
   LayerProbe(const char * filename, HyPerCol * hc);
   virtual ~LayerProbe();

   virtual int outputState(float time, HyPerLayer * l) = 0;

protected:
   FILE * fp;
};

}

#endif /* LAYERPROBE_HPP_ */
