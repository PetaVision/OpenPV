/*
 * CustomGroupHandler.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "NormalizeL3.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace PV {

CustomGroupHandler::CustomGroupHandler() {}

CustomGroupHandler::~CustomGroupHandler() {}

ParamGroupType CustomGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (!strcmp(keyword, "normalizeL3")) {
      result = WeightNormalizerGroupType;
   }
   return result;
}

NormalizeBase * CustomGroupHandler::createWeightNormalizer(char const * keyword, char const * name, HyPerCol * hc) {
   NormalizeBase * normalizer = NULL;
   if (!strcmp(keyword, "normalizeL3")) {
      normalizer = new NormalizeL3(name, hc);
      if (normalizer == NULL) {
         fprintf(stderr, "Error creating %s \"%s\".\n", keyword, name);
         exit(EXIT_FAILURE);
      }
   }
   return normalizer;
}

}  // namespace PV


