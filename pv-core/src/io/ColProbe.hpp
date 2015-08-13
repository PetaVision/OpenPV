/*
 * ColProbe.hpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#ifndef COLPROBE_HPP_
#define COLPROBE_HPP_

#include <string.h>
#include "../columns/HyPerCol.hpp"

namespace PV {

/**
 * ColProbe is the base class for probes that are attached to the column as a whole,
 * as opposed to an individual layer or connection.
 *
 * The original motivation for ColProbe was for computing total energy of a sparse-coding
 * hierarchy.  In this situation, the energy is a sum of contributions from the residual
 * layer and the sparse representation layer, and we need the energy for each element
 * of the batch.  The getValues() method would compute the energy for each element of the
 * batch.  The getValue() method returns the energy for a single batch element.
 *
 * A HyPerCol object with dtAdaptFlag set to true uses a ColProbe::getValues() call to
 * compute the dtAdapt vector.
 */
class ColProbe {
public:
   /**
    * Public constructor for the ColProbe class.
    */
   ColProbe(const char * probeName, HyPerCol * hc);

   /**
    * Destructor for the ColumnEnergyProbe class.
    */
   virtual ~ColProbe();

   /**
    * Reads parameters from the params file/writes parameters to the output params file.
    * Derived classes should not override or hide this method.  Instead, they should
    * override the protected method ioParamsFillGroup(), which is called by ioParams().
    */
   int ioParams(enum ParamsIOFlag ioFlag);
   
   /**
    * The virtual method for outputting the quantities measured by the ColProbe.
    * Derived classes should override this method.  Typically, outputState
    * will fprintf to stream->fp, where stream is the ColProbe member variable.
    */
   virtual int outputState(double time, HyPerCol * hc) {return PV_SUCCESS;}
   
   /**
    * Public get-method for returning the name of the ColProbe, which is set during
    * initialization.
    */
   char const * getColProbeName() { return colProbeName; }
   
   /**
    * Searches the params database belonging to the HyPerCol specified during initialization
    * for the group whose name is the probe's name, and returns the associated keyword
    * (typically "ColProbe" or the name of the derived class).
    */
   char const * keyword();
    
   /**
    * Derived classes of ColProbe should override this method to return a vector of length
    * getVectorSize().
    */
   virtual int getValues(double timevalue, std::vector<double> * values);
   
   /**
    * Derived classes of ColProbe should override getValue() to
    * return a value for one index in the range 0, 1, ..., getVectorSize()-1.
    */
   virtual double getValue(double timevalue, int index);
   
   /**
    * Derived classes of ColProbe should override getVectorSize()
    * to return the size of the vectors that getValues computes.
    */
   size_t getVectorSize() { return 0; }

protected:
   HyPerCol * parentCol;
   PV_Stream * stream;
   char * colProbeName;

   /**
    * The constructor without arguments should be used by derived classes.
    */
   ColProbe();

   /**
    * Reads the parameters and performs initializations that do not
    * depend on other param groups.  It is called by the public constructor
    * and should be called by the initializer of any derived classes.
    */
   int initialize(const char * probeName, HyPerCol * hc);

   /**
    * Reads parameters from the params file/writes parameters to the output params file.
    * If a derived class introduces a new parameter, its ioParamsFillGroup method should
    * call an ioParam_ method for that parameter.  If a derived class eliminates a
    * a parameter, or changes the dependencies of the parameter, it should override
    * the ioParam_ method for that parameter.  The derived class's ioParamsFillGroup
    * method should call its base class's ioParamsFillGroup method.
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /**
    * @brief probeOutputFile: the path, relative to the HyPerCol's outputPath directory,
    * to write to during outputState.
    * @details If blank, output is sent to stdout.
    */
   virtual void ioParam_probeOutputFile(enum ParamsIOFlag ioFlag);
    
   /**
    * Opens a stream for writing associated to the given path.  Relative paths are
    * relative to the HyPerCol's outputPath.
    * Under MPI, only the root process initializes the stream; nonroot processes set
    * the stream member variable to NULL.
    * This method is called by ColProbe::ioParam_probeOutputFile during initialization.
    * It is an error to call it again once the stream has been initialized.
    */
   int initialize_stream(const char * filename);
    
   /**
    * Called by initialize_stream after opening the stream member variable.
    * Derived classes can override this method to write header data to the output
    * file.
    */
   virtual int outputHeader() { return PV_SUCCESS; }

private:
    
    /**
     * Sets the colProbeName member variable.  It is called by ColProbe::initialize().
     */
   int setColProbeName(const char * name);
    
   /**
    * Sets member variables to safe values.  It is called by both the
    * public and protected constructors, and should not otherwise be called.
    */
   int initialize_base();
}; // end class ColProbe

}  // end namespace PV

#endif /* COLPROBE_HPP_ */
