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

class ColumnEnergyProbe : public ColProbe {
public:
   ColumnEnergyProbe(const char * probename, HyPerCol * hc);
   ~ColumnEnergyProbe();

   
   /** @brief Adds a probe to the energy calculation.
    * @details Returns PV_SUCCESS if the probe is added successfully.
    * If probe is NULL, the list of terms is unchanged and PV_FAILURE is returned.
    * Nothing prevents a probe from being added more than once.
    */
   int addTerm(BaseProbe * probe, double coefficient, size_t vectorSize);
   virtual int outputState(double timevalue, HyPerCol * hc);
   
   virtual int getValues(double timevalue, std::vector<double> * values);
   
   /**
    * Computes the total energy based on the list of probes
    * that have been added using addTerm().
    * For each n, the energy is increased
    * by terms[n].coeff * terms[n].probe->getValue().
    */
   virtual double getValue(double timevalue, int index);

protected:
   ColumnEnergyProbe();
   int initializeColumnEnergyProbe(const char * probename, HyPerCol * hc);
   virtual int outputHeader();

   size_t numTerms;
   energyTerm * terms;
   size_t vectorSize;

private:
   int initialize_base();

}; // end class ColumnEnergyProbe

}  // end namespace PV

#endif /* COLUMNENERGYPROBE_HPP_ */
