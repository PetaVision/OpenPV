/*
 * ColProbe.hpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#ifndef COLPROBE_HPP_
#define COLPROBE_HPP_

#include "BaseProbe.hpp"
#include <string.h>

namespace PV {

class HyPerCol;

/**
 * ColProbe is the base class for probes that are attached to the column as a
 * whole,
 * as opposed to an individual layer or connection.
 * Derived classes must implement the needRecalc and calcValues methods.
 *
 *
 * The original motivation for ColProbe was for computing total energy of a
 * sparse-coding
 * hierarchy.  In this situation, the energy is a sum of contributions from the
 * residual
 * layer and the sparse representation layer, and we need the energy for each
 * element
 * of the batch.  The getValues() method would compute the energy for each
 * element of the
 * batch.  The getValue() method returns the energy for a single batch element.
 *
 * An AdaptiveTimeScaleProbe with targetName set to a ColProbe
 * uses a ColProbe::getValues() call to compute the timeScaleTrue vector.
 */
class ColProbe : public BaseProbe {
  public:
   /**
    * Public constructor for the ColProbe class.
    */
   ColProbe(const char *name, HyPerCol *hc);

   /**
    * Destructor for the ColProbe class.
    */
   virtual ~ColProbe();

   virtual Response::Status respond(std::shared_ptr<BaseMessage const> message) override;

   /**
    * Calls BaseProbe::communicateInitInfo (which sets up any triggering or
    * attaching to an energy probe)
    */
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * The virtual method for outputting the quantities measured by the ColProbe.
    * Derived classes should override this method.  Typically, outputState
    * will fprintf to outputstream->fp, where stream is the BaseProbe member
    * variable.
    */
   virtual Response::Status outputState(double timed) override { return Response::SUCCESS; }

  protected:
   /**
    * The constructor without arguments should be used by derived classes.
    */
   ColProbe();

   /**
    * Reads the parameters and performs initializations that do not
    * depend on other param groups.  It is called by the public constructor
    * and should be called by the initializer of any derived classes.
    */
   int initialize(const char *name, HyPerCol *hc);

   /**
    * Reads parameters from the params file/writes parameters to the output
    * params file.
    * If a derived class introduces a new parameter, its ioParamsFillGroup method
    * should
    * call an ioParam_ method for that parameter.  If a derived class eliminates
    * a
    * a parameter of the base class, or changes the dependencies of a parameter,
    * it
    * should override the ioParam_ method for that parameter.  The derived
    * class's
    * ioParamsFillGroup method should call its base class's ioParamsFillGroup
    * method.
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief targetName: ColProbe sets targetName to the empty string,
    * since the only possible target is the parent HyPerCol. The targenName
    * parameter is neither read nor written by this method.
    */
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag) override;

   Response::Status
   respondColProbeWriteParams(std::shared_ptr<ColProbeWriteParamsMessage const>(message));
   Response::Status
   respondColProbeOutputState(std::shared_ptr<ColProbeOutputStateMessage const>(message));

   /**
    * Calls BaseProbe::initOutputStreams and then calls outputHeader()
    */
   virtual void initOutputStreams(const char *filename, Checkpointer *checkpointer) override;

   /**
    * Called by initialize_stream after opening the stream member variable.
    * Derived classes can override this method to write header data to the output
    * file.
    */
   virtual void outputHeader() {}

  private:
   /**
    * Initializes member variables to safe values (e.g. pointers are set to
    * NULL).
    * It is called by both the public and protected constructors, and should not
    * otherwise be called.
    */
   int initialize_base();
}; // end class ColProbe

} // end namespace PV

#endif /* COLPROBE_HPP_ */
