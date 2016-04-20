/*
 * QuotientColProbe.hpp
 *
 *  Created on: Aug 12, 2015
 *      Author: pschultz
 */

#ifndef QUOTIENTCOLPROBE_HPP_
#define QUOTIENTCOLPROBE_HPP_

#include "ColProbe.hpp"

namespace PV {

class BaseProbe;

/**
 * QuotientColProbe takes two probes (of any type) and reports their quotient.
 * The original motivation for QuotientColProbe was to get total energy of a sparse-coding
 * module scaled by the size of the input image.
 * Note that this probe depends on other probes and that there is NO checking to
 * prevent loops in the probe dependencies.
 */

class QuotientColProbe : public ColProbe {
public:
   /**
    * Public constructor for the QuotientColProbe class.
    */
   QuotientColProbe(const char * probename, HyPerCol * hc);
   
   /**
    * Destructor for the QuotientColProbe class.
    */
   virtual ~QuotientColProbe();

   /**
    * Calls ColProbe::ioParamsFillGroup and then reads/writes the parameters added by QuotientColProbe
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   
   /**
    * List of QuotientColProbe parameters
    * @name QuotientColProbe Parameters
    * @{
    */
   
   /**
    * @brief valueDescription: a short description of what the quantities computed by getValues() represent.
    * @details when outputHeader is called, it prints a line to the output file
    * consisting of the string "Probe_name,time,index," followed by the valueDescription.
    * Defaults to "value".
    */
   virtual void ioParam_valueDescription(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief numerator: A probe whose values are used in the numerators of the quotients.
    * numerator can be a layer probe, a connection probe, or a column probe.
    * @details numerator->getNumValues() and denominator->getNumValues() must return the same value,
    * which then becomes the value returned by the QuotientColProbe's getNumValues().
    */
   virtual void ioParam_numerator(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief denominator: A probe whose values are used in the denominators of the quotients.
    * denominator can be a layer probe, a connection probe, or a column probe.
    * @details numerator->getNumValues() and denominator->getNumValues() must return the same value,
    * which then becomes the value returned by the QuotientColProbe's getNumValues().
    */
   virtual void ioParam_denominator(enum ParamsIOFlag ioFlag);

   /** @} */ /* end of io functions for QuotientColProbe parameters */
   
   virtual int communicateInitInfo();
   
   /**
    * A function to find a probe, whether it belongs to a layer, a connection, or the hypercol.
    * Returns NULL if the probe cannot be found.
    */
   BaseProbe * findProbe(char const * probeName);
   
   /**
    * Prints the energies to the output stream, formatted as a comma-separated value:
    * "Name of probe",timevalue,index,energy
    * The number of lines printed is equal to getVectorSize(), and index goes from 0 to getVectorSize()-1.
    */
   virtual int outputState(double timevalue);

protected:
   /**
    * The constructor without arguments should be used by derived classes.
    */
   QuotientColProbe();
   
   /**
    * Reads the parameters and performs initializations that do not
    * depend on other param groups.  It is called by the public constructor
    * and should be called by the initializer of any derived classes.
    */
   int initializeQuotientColProbe(const char * probename, HyPerCol * hc);
   
   virtual bool needRecalc(double timevalue) { return true; }
   
   virtual double referenceUpdateTime() const;

   /**                                                                             
    * Implements the needRecalc method.  Always returns true, in the expectation
    * that the hard work is done by the probes in the numerator and denominator. 
    */                                                                             
   virtual int calcValues(double timeValue);
   
   virtual int outputHeader();

private:
   /**
    * Sets member variables to safe values.  It is called by both the
    * public and protected constructors, and should not otherwise be called.
    */
   int initialize_base();

// Member variables
protected:
   char * valueDescription; // A string description of the quantity calculated by the probe, used by outputHeader
   char * numerator; // The name of the probe that supplies the numerator
   char * denominator; // The name of the probe that supplies the denominator
   BaseProbe * numerProbe; // A pointer to the probe that supplies the numerator.
   BaseProbe * denomProbe; // A pointer to the probe that supplies the denominator.

}; // end class QuotientColProbe

BaseObject * createQuotientColProbe(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* QUOTIENTCOLPROBE_HPP_ */
