/*
 * CustomParamGroupHandler.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: peteschultz
 */

#include "CustomParamGroupHandler.hpp"
#include "FirmThresholdCostFnProbe.hpp"
#include "L0NormProbe.hpp"
#include "L1NormProbe.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

CustomParamGroupHandler::CustomParamGroupHandler() {
}

CustomParamGroupHandler::~CustomParamGroupHandler() {
}

ParamGroupType CustomParamGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result;
   if (keyword==NULL) { result = UnrecognizedGroupType; }
   else if (!strcmp("FirmThresholdCostFnProbe", keyword)) {
       result = ProbeGroupType;
   }
   else if (!strcmp("L0NormProbe", keyword)) {
       result = ProbeGroupType;
   }
   else if (!strcmp("L1NormProbe", keyword)) {
       result = ProbeGroupType;
   }
   else {
       result = UnrecognizedGroupType;
   }
   return result;
}

BaseProbe * CustomParamGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   BaseProbe * addedProbe = NULL;

   // Layer probe keywords
   if (keyword==NULL) {
      addedProbe = NULL;
   }
   else if( !strcmp(keyword, "FirmThresholdCostFnProbe") ) {
      addedProbe = new FirmThresholdCostFnProbe(name, hc);
   }
   else if( !strcmp(keyword, "L0NormProbe") ) {
      addedProbe = new L0NormProbe(name, hc);
   }
   else if( !strcmp(keyword, "L1NormProbe") ) {
      addedProbe = new L1NormProbe(name, hc);
   }

   if (addedProbe==NULL && getGroupType(keyword)==ProbeGroupType) {
         if (hc->columnId()==0) {
            fprintf(stderr, "createProbe error: unable to add %s\n", keyword);
         }
         MPI_Barrier(hc->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
   }

   return addedProbe;
}

} /* namespace PV */
