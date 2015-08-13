/*
 * ColumnEnergyProbe.hpp
 *
 *  Created on: Aug 12, 2015
 *      Author: pschultz
 */

#ifndef COLUMNENERGYPROBE_HPP_
#define COLUMNENERGYPROBE_HPP_

#include "ColProbe.hpp"

namespace PV {

class BaseProbe;

typedef struct energyterm_ {
   BaseProbe * probe;
   double coeff;
} energyTerm;

/**
 * ColumnEnergyProbe assembles several base probes each of which
 * contribute a term to an energy of the entire HyPerCol.
 *
 * At the params file level, define a ColumnEnergyProbe group.  Say,
 * for example that the group is named "total_energy".  Then for each base
 * probe you want to add to the probe, its params should include the line
 *
 * energyProbe = "total_energy";
 *
 * Then optionally specify a coefficient, which defaults to 1.
 *
 * The contribution from this probe to the total energy is
 * coefficient * value
 * where value is the result of calling the BaseProbe's getValues() method.
 *
 * At the C/C++ code level, BaseProbes register themselves to the
 * ColumnEnergyProbe by calling the ColumnEnergyProbe's addTerm() method.
 * Each call to a ColumnEnergyProbe's object that calls addTerm() must
 * use the same vectorSize value, which must agree with the size of the
 * vector computed by the BaseProbe's getValues() method.
 */

class ColumnEnergyProbe : public ColProbe {
public:
   /**
    * Public constructor for the ColumnEnergyProbe class.
    */
   ColumnEnergyProbe(const char * probename, HyPerCol * hc);
   
   /**
    * Destructor for the ColumnEnergyProbe class.
    */
   virtual ~ColumnEnergyProbe();

   /** @brief Adds a probe to the energy calculation.
    * @details Returns PV_SUCCESS if the probe is added successfully.
    * If probe is NULL, the list of terms is unchanged and PV_FAILURE is returned.
    * Nothing prevents a probe from being added more than once.
    * All BaseProbes added to the ColumnEnergyProbe must pass the same
    * vectorSize. Their getValues() method must return a vector of
    * size vectorSize, and their getValue method must return a value
    * for any index with 0 <= index < vectorSize.
    */
   int addTerm(BaseProbe * probe, double coefficient, size_t vectorSize);
   
   /**
    * Prints the energies to the output stream, formatted as a comma-separated value:
    * "Name of probe",timevalue,index,energy
    * The number of lines printed is equal to getVectorSize(), and index goes from 0 to getVectorSize()-1.
    */
   virtual int outputState(double timevalue, HyPerCol * hc);
   
   /**
    * Computes the vector of total energies.  Any existing contents of *values
    * are clobbered.  On return, *values is a vector of length
    * getVectorSize(), and values[b] is the energy for index b.
    */
   virtual int getValues(double timevalue, std::vector<double> * values);
   
   /**
    * Returns the total energy for the given index.
    */
   virtual double getValue(double timevalue, int index);
   
   /**
    * Returns the vectorSize determined by calls to addTerm().
    * All calls to addTerm must pass the same value of vectorSize.
    */
   size_t getVectorSize() { return vectorSize; }

protected:
   /**
    * The constructor without arguments should be used by derived classes.
    */
   ColumnEnergyProbe();
   
   /**
    * Reads the parameters and performs initializations that do not
    * depend on other param groups.  It is called by the public constructor
    * and should be called by the initializer of any derived classes.
    */
   int initializeColumnEnergyProbe(const char * probename, HyPerCol * hc);
   virtual int outputHeader();

   size_t numTerms;
   energyTerm * terms;
   size_t vectorSize;

private:
   /**
    * Sets member variables to safe values.  It is called by both the
    * public and protected constructors, and should not otherwise be called.
    */
   int initialize_base();

}; // end class ColumnEnergyProbe

}  // end namespace PV

#endif /* COLUMNENERGYPROBE_HPP_ */
