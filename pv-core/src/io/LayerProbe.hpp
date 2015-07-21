/*
 * LayerProbe.h
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#ifndef LAYERPROBE_HPP_
#define LAYERPROBE_HPP_

#include <stdio.h>
#include "../io/fileio.hpp"
#include "BaseProbe.hpp"

namespace PV {

class HyPerCol;
class HyPerLayer;

typedef enum {
   BufV,
   BufActivity
} PVBufType;

class LayerProbe : public BaseProbe {

// Methods
public:
   LayerProbe(const char * probeName, HyPerCol * hc);
   virtual ~LayerProbe();

   virtual int communicateInitInfo();

   HyPerLayer * getTargetLayer() {return targetLayer;}

protected:
   LayerProbe();
   int initialize(const char * probeName, HyPerCol * hc);
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();
   int setTargetLayer(const char * layerName);

// Member variables
protected:
   HyPerLayer * targetLayer;

};

}

#endif /* LAYERPROBE_HPP_ */
