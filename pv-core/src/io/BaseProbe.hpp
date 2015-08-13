/*
 * BaseProbe.h
 *
 *      Author: slundquist
 */

#ifndef BASEPROBE_HPP_
#define BASEPROBE_HPP_

#include <stdio.h>
#include <vector>
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

   //virtual and necessary to overwrite for attaching to target layer or connection
   virtual int communicateInitInfo() = 0;
   //virtual, but not nessessary to overwrite
   virtual int allocateDataStructures();

   virtual bool needUpdate(double time, double dt);
   virtual int outputStateWrapper(double timef, double dt);
   virtual int outputState(double timef) = 0;
   virtual int writeTimer(FILE* stream) {return PV_SUCCESS;}

   const char * getName() {return name;}
   const char * getTargetName() {return targetName;}
   void const * getOwner() { return owner;}
   
   /**
    * getValues() sets the vector argument to the values of the probe.
    * BaseProbe::getValues() always leaves the vector untouched and always returns PV_FAILURE.
    * derived classes should override this method.  The number of elements of the vector is up to
    * the derived class.
    * getValues() was motivated by the need to have a layer report its energy
    * for each element of the batch.  In this case the number of elements
    * of the vector is the HyPerCol's batch size.
    * 
    */
   virtual int getValues(double timevalue, std::vector<double> * values) { return PV_FAILURE; }
   /**
    * getValue() is meant for situations where the caller needs one value
    * of the vector that would be returned by getValues(), not the whole vector.
    * the base class always returns zero, no matter the value of the index.
    * Derived classes should override this method.
    */
   virtual double getValue(double timevalue, int index) { return 0.0; }

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
   virtual void ioParam_energyProbe(enum ParamsIOFlag ioFlag);
   virtual void ioParam_coefficient(enum ParamsIOFlag ioFlag);
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
   void * owner; // the object responsible for calling the probe's destructor
   char * name;
   char * targetName;
   char * energyProbe; // the name of the ColumnEnergyProbe to attach to, if any.
   double coefficient;

private:
   char * msgparams; // the message parameter in the params
   char * msgstring; // the string that gets printed by outputState ("" if message is empty or null; message + ":" if nonempty
   char * probeOutputFilename;
};

}

#endif /* BASEPROBE_HPP_ */
