/*
 * PVLayerProbe.h
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#ifndef PVLAYERPROBE_HPP_
#define PVLAYERPROBE_HPP_

#include "../layers/PVLayer.h"

#include <stdio.h>

namespace PV {

typedef enum {
   BufV,
   BufActivity
} PVBufType;

class PVLayerProbe {
public:
   PVLayerProbe();
   PVLayerProbe(const char * filename);
   virtual ~PVLayerProbe();

   virtual int outputState(float time, PVLayer * l) = 0;

protected:
   FILE * fp;
};

}

#endif /* PVLAYERPROBE_HPP_ */
