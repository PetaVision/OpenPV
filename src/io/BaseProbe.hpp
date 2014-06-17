/*
 * BaseProbe.h
 *
 *      Author: slundquist
 */

#ifndef BASEPROBE_HPP_
#define BASEPROBE_HPP_

#include <stdio.h>
#include "../io/fileio.hpp"

namespace PV {

class HyPerCol;
class HyPerLayer;

//typedef enum {
//   BufV,
//   BufActivity
//} PVBufType;

class BaseProbe {

// Methods
public:
   //BaseProbe(const char * probeName, HyPerCol * hc);
   virtual ~BaseProbe();

   int ioParams(enum ParamsIOFlag ioFlag);

   //virtual and necessary to overwrite for attaching to target layerse
   virtual int communicateInitInfo() = 0;
   //virtual, but not nessessary to overwrite
   virtual int allocateDataStructures();

   virtual bool needUpdate(double time, double dt);
   virtual int outputStateWrapper(double timef, double dt);
   virtual int outputState(double timef) = 0;
   virtual int checkpointTimers(PV_Stream * timerstream) {return PV_SUCCESS;}

   const char * getName() {return name;}
   const char * getTargetName() {return targetName;}

protected:
   BaseProbe();
   int initialize(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_message(enum ParamsIOFlag ioFlag);
   virtual void ioParam_probeOutputFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);
   virtual int initOutputStream(const char * filename);

   HyPerCol * getParent() {return parent;}
   const char * getMessage() {return msgstring;}
   virtual int initMessage(const char * msg);
   PV_Stream * getStream() {return outputstream;}

private:
   int initialize_base();
   void setParentCol(HyPerCol * hc) {parent = hc;}
   int setProbeName(const char * probeName);

// Member variables
protected:
   PV_Stream * outputstream;
   bool triggerFlag;
   char* triggerLayerName;
   HyPerLayer * triggerLayer;
   double triggerOffset;
   HyPerCol * parent;
   char * name;
   char * targetName;

private:
   char * msgparams; // the message parameter in the params
   char * msgstring; // the string that gets printed by outputState ("" if message is empty or null; message + ":" if nonempty
   char * probeOutputFilename;
};

}

#endif /* BASEPROBE_HPP_ */
