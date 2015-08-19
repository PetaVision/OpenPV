/*
 * ConnFunctionProbe.cpp
 *
 *  Created on: Mar 23, 2012
 *      Author: pschultz
 */

#include "ConnFunctionProbe.hpp"

namespace PV {

ConnFunctionProbe::ConnFunctionProbe()
{
   initialize_base();
}

ConnFunctionProbe::ConnFunctionProbe(const char * probename, HyPerCol * hc) {
   initialize_base();
   initialize(probename, hc);
}

int ConnFunctionProbe::initialize_base() {
   parentGenColName = NULL;
   parentGenCol = NULL;
   return PV_SUCCESS;
}

int ConnFunctionProbe::initialize(const char * probename, HyPerCol * hc) {
   int status = BaseConnectionProbe::initialize(probename, hc);
   return status;
}

int ConnFunctionProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseConnectionProbe::ioParamsFillGroup(ioFlag);
   ioParam_parentGenColProbe(ioFlag);
   return status;
}

void ConnFunctionProbe::ioParam_parentGenColProbe(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "parentGenColProbe", &parentGenColName, NULL);
   // There doesn't need to be a parent GenColProbe, so it's not an error if this returns null.
}

int ConnFunctionProbe::communicate() {
   int status = PV_SUCCESS;
   if( parentGenColName != NULL ) {
      ColProbe * parent_col = parent->getColProbeFromName(parentGenColName);
      if (parent_col != NULL) {
         parentGenCol = dynamic_cast<GenColProbe *>(parent_col);
         if( parentGenCol == NULL) {
            fprintf(stderr, "ConnFunctionProbe \"%s\": parentGenColProbe \"%s\" is not a GenColProbe\n", name, parentGenColName);
            status = PV_FAILURE;
         }
         else {
            parentGenCol->addConnTerm(this, targetConn, 1.0f);
            // ReciprocalEnergyProbe uses reciprocalFidelityCoeff for the energy
            // so I think a separate probe parameter for the coefficient won't be necessary.
         }
      }
   }
   return status;
}

int ConnFunctionProbe::allocateProbe() {
   return PV_SUCCESS;
}

int ConnFunctionProbe::outputState(double timef) {
   int status = PV_SUCCESS;
   if( status == PV_SUCCESS ) {
      double energy = evaluate(timef);
      fprintf(getStream()->fp, "Time %f: Energy %f\n", timef, energy);
   }
   return status;
}

ConnFunctionProbe::~ConnFunctionProbe()
{
}

}  // end namespace PV
