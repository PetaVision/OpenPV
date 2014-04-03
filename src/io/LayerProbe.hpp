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
   LayerProbe(const char * probeName, HyPerCol * hc);
   virtual ~LayerProbe();

   int ioParams(enum ParamsIOFlag ioFlag);

   virtual int communicateInitInfo();

   virtual bool needUpdate(double time, double dt);
   virtual int outputStateWrapper(double timef, double dt);
   virtual int outputState(double timef) = 0;

   const char * getProbeName() {return probeName;}
   HyPerLayer * getTargetLayer() {return targetLayer;}

protected:
   LayerProbe();
   int initLayerProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_targetLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_message(enum ParamsIOFlag ioFlag);
   virtual void ioParam_probeOutputFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);
   virtual int initOutputStream(const char * filename);

   HyPerCol * getParentCol() {return parentCol;}
   const char * getMessage() {return msgstring;}
   virtual int initMessage(const char * msg);

private:
   int initLayerProbe_base();
   void setParentCol(HyPerCol * hc) {parentCol = hc;}
   int setProbeName(const char * probeName);
   int setTargetLayer(const char * layerName);

// Member variables
protected:
   PV_Stream * outputstream;
   bool triggerFlag;
   char* triggerLayerName;
   HyPerLayer * triggerLayer;
   double triggerOffset;
   HyPerCol * parentCol;
   char * probeName;
   HyPerLayer * targetLayer;

private:
   char * targetLayerName;
   char * msgparams; // the message parameter in the params
   char * msgstring; // the string that gets printed by outputState ("" if message is empty or null; message + ":" if nonempty
   char * probeOutputFilename;
};

}

#endif /* LAYERPROBE_HPP_ */
