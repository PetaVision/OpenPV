/*
 * ConvertFromTable.hpp
 *
 *  Created on: Dec 1, 2015
 *      Author: pschultz
 */

#ifndef SRC_CONVERTFROMTABLE_HPP_
#define SRC_CONVERTFROMTABLE_HPP_

#include <layers/CloneVLayer.hpp>


struct convTableStruct_ {
   int numPoints;
   int numFeatures;
   float minRecon;
   float maxRecon;
};
typedef struct convTableStruct_ convTableStruct;

class ConvertFromTable: public PV::CloneVLayer {
public:
   ConvertFromTable(char const * name, PV::HyPerCol * hc);
   int allocateDataStructures();
   virtual ~ConvertFromTable();

protected:
   ConvertFromTable();
   int initialize(char const * name, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_dataFile(enum PV::ParamsIOFlag ioFlag);
   virtual int updateState(double timed, double dt);

   int loadConversionTable();

private:
   int initialize_base();

// Member variables
protected:
   char * dataFile;
   convTableStruct convTable;
   float * convData;
}; // class createConvertFromTable

PV::BaseObject * createConvertFromTable(char const * name, PV::HyPerCol * hc);

#endif /* SRC_CONVERTFROMTABLE_HPP_ */
