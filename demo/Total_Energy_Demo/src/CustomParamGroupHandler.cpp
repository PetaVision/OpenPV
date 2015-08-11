/*
 * CustomParamGroupHandler.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: peteschultz
 */

#include "CustomParamGroupHandler.hpp"
#include "L1NormProbe.hpp"

namespace PV {

CustomParamGroupHandler::CustomParamGroupHandler() {
}

CustomParamGroupHandler::~CustomParamGroupHandler() {
}

ParamGroupType CustomParamGroupHandler::getGroupType(char const * keyword) {
   struct keyword_grouptype_entry  {char const * kw; ParamGroupType type;};
   struct keyword_grouptype_entry keywordtable[] = {
         // Probes
         // // Layer probes
         {"L1NormProbe", ProbeGroupType},
         {NULL, UnrecognizedGroupType}
   };
   ParamGroupType result;
   if (keyword==NULL) { result = UnrecognizedGroupType; }
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
