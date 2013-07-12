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
   BaseConnectionProbe::initialize(probename, hc);
   const char * parent_col_name = hc->parameters()->stringValue(name, "parentGenColProbe");
   // There doesn't need to be a parent GenColProbe, so it's not an error if this returns null.
   if (parent_col_name != NULL) {
      parentGenColName = strdup(parent_col_name);
      if (parentGenColName==NULL) {
         fprintf(stderr, "ConnFunctionProbe error: unable to allocate memory for name of parent GenColProbe.\n");
         exit(EXIT_FAILURE);
      }
   }
   return PV_SUCCESS;
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
