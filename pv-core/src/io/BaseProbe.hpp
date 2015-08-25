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

/**
 * An abstract base class for the common functionality of layer probes and connection probes.
 */
class BaseProbe {

// Methods
public:
   //BaseProbe(const char * probeName, HyPerCol * hc);
   virtual ~BaseProbe();

   int ioParams(enum ParamsIOFlag ioFlag);


   /**
    * A pure virtual function called during HyPerCol::run, during the communicateInitInfo stage.
    * BaseProbe::communicateInitInfo sets up the triggering layer and attaches to the energy probe,
    * if either triggerFlag or energyProbe are set.
    */
   virtual int communicateInitInfo() = 0;

   /**
    * Called during HyPerCol::run, during the allocateDataStructures stage.
    * BaseProbe::allocateDataStructures returns immediately, but derived classes that
    * need to allocate storage should override this method.
    */
   virtual int allocateDataStructures();

   /**
    * If there is a triggering layer, needUpdate returns true when the triggering layer's
    * nextUpdateTime, modified by the probe's triggerOffset parameter, occurs.
    * If there is not a triggering layer, needUpdate returns false.
    * This behavior can be overridden if a probe uses some criterion other than triggering
    * to choose when output its state.
    */
   virtual bool needUpdate(double time, double dt);
   
   /**
    * The public interface for calling the outputState method.
    * BaseConnection::outputStateWrapper calls outputState() if needUpdate() returns true.
    * This behavior is intended to be general, but the method can be overridden if needed.
    */
   virtual int outputStateWrapper(double timef, double dt);
   
   /**
    * A pure virtual method for writing output to the output file.
    */
   virtual int outputState(double timef) = 0;
   virtual int writeTimer(FILE* stream) {return PV_SUCCESS;}

   /**
    * Returns the keyword of the params group associated with this probe.
    */
   char const * getKeyword();

   /**
    * Returns the name of the probe, specified in the public constructor.
    */
   const char * getName() {return name;}
   
   /**
    * Returns the name of the targetName parameter for this probe.
    * LayerProbe uses targetName to specify the layer to attach to;
    * BaseConnectionProbe uses it to specify the connection to attach to.
    */
   const char * getTargetName() {return targetName;}
   
   /**
    * Specifies the object responsible calling the probe's destructor.
    * BaseProbe sets owner to the parent HyPerCol during initialization.
    * During the communicateInitInfo stage, layer probes and connection probes
    * change their owner to the layer or connection they attach to.
    */
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

   /** 
    * List of parameters for the BaseProbe class
    * @name BaseProbe Parameters
    * @{
    */

   /**
    * @brief targetName: the name of the object that the probe attaches to.
    * In LayerProbe, targetName is used to define the targetLayer, and in
    * BaseConnectionProbe, targetName is used to define the targetConn.
    */
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief message: A string parameter that is typically included in the lines output by the outputState method
    */
   virtual void ioParam_message(enum ParamsIOFlag ioFlag);

   /**
    * @brief probeOutputFile: the name of the file that the outputState method writes to.
    * If blank, the output is sent to stdout.
    */
   virtual void ioParam_probeOutputFile(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerFlag: If false, the needUpdate method always returns true,
    * so that outputState is called every timestep.  If false, the needUpdate
    * method uses triggerLayerName and triggerOffset to determine if the probe
    * should be triggered.
    */
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief triggerLayerName: If triggerFlag is true, triggerLayerName specifies the layer
    * to check for triggering.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief triggerOffset: If triggerFlag is true, triggerOffset specifies the
    * time interval *before* the triggerLayer's nextUpdate time that needUpdate() returns true.
    */
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief energyProbe: If nonblank, specifies the name of a ColumnEnergyProbe
    * that this probe contributes an energy term to.
    */
   virtual void ioParam_energyProbe(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief coefficient: If energyProbe is set, the coefficient parameter specifies
    * that ColumnEnergyProbe multiplies the result of this probe's getValues() method
    * by coefficient when computing the error.
    * @details Note that coefficient does not affect the value returned by the getValue() or
    * getValues() method.
    */
   virtual void ioParam_coefficient(enum ParamsIOFlag ioFlag);
   /** @} */

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
