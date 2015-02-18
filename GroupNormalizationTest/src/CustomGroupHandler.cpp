/*
 * CustomGroupHandler.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "AllConstantValueProbe.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace PV {

CustomGroupHandler::CustomGroupHandler() {}

CustomGroupHandler::~CustomGroupHandler() {}

ParamGroupType CustomGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (!strcmp(keyword, "AllConstantValueProbe")) {
      result = ProbeGroupType;
   }
   return result;
}

BaseProbe * CustomGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   BaseProbe * probe = NULL;
   if (!strcmp(keyword, "AllConstantValueProbe")) {
      probe = new AllConstantValueProbe(name, hc);
      if (probe == NULL) {
         fprintf(stderr, "Error creating %s \"%s\".\n", keyword, name);
         exit(EXIT_FAILURE);
      }
   }
   return probe;
}


}  // namespace PV


