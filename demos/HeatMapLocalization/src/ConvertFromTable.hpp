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
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dataFile(enum ParamsIOFlag ioFlag);
   virtual int doUpdateState(double timed, double dt, const PVLayerLoc * loc,
         pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * GSynHead);

   int loadConversionTable();

private:
   int initialize_base();

// Member variables
protected:
   char * dataFile;
   convTableStruct convTable;
   float * convData;
};

#endif /* SRC_CONVERTFROMTABLE_HPP_ */
