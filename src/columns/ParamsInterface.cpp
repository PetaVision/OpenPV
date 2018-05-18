/*
 * ParamsInterface.cpp
 *
 *  Created on May 16, 2018
 *      Author: Pete Schultz
 */

#include "ParamsInterface.hpp"

namespace PV {
ParamsInterface::~ParamsInterface() { free(name); }

int ParamsInterface::initialize(char const *name, PVParams *params) {
   CheckpointerDataInterface::initialize();
   setName(name);
   setParams(params);
   setObjectType();
   setDescription(getObjectType() + " \"" + getName() + "\"");
   return PV_SUCCESS;
}

void ParamsInterface::setName(char const *name) {
   pvAssert(this->name == nullptr);
   this->name = strdup(name);
   FatalIf(name == nullptr, "could not set name \"%s\". %s\n", name, strerror(errno));
}

void ParamsInterface::setParams(PVParams *params) { mParams = params; }

void ParamsInterface::setObjectType() {
   mObjectType = parameters()->groupKeywordFromName(getName());
}

void ParamsInterface::ioParams(enum ParamsIOFlag ioFlag, bool printHeader, bool printFooter) {
   if (printHeader) {
      ioParamsStartGroup(ioFlag);
   }
   ioParamsFillGroup(ioFlag);
   if (printFooter) {
      ioParamsFinishGroup(ioFlag);
   }
}

void ParamsInterface::ioParamsStartGroup(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_WRITE) {
      const char *keyword = mParams->groupKeywordFromName(getName());

      auto *printParamsStream = mParams->getPrintParamsStream();
      if (printParamsStream) {
         printParamsStream->printf("\n");
         printParamsStream->printf("%s \"%s\" = {\n", keyword, getName());
      }

      auto *printLuaStream = mParams->getPrintLuaStream();
      if (printLuaStream) {
         printLuaStream->printf("%s = {\n", getName());
         printLuaStream->printf("groupType = \"%s\";\n", keyword);
      }
   }
}

void ParamsInterface::ioParamsFinishGroup(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_WRITE) {
      auto *printParamsStream = mParams->getPrintParamsStream();
      if (printParamsStream) {
         printParamsStream->printf("};\n");
      }

      auto *printLuaStream = mParams->getPrintLuaStream();
      if (printLuaStream) {
         printLuaStream->printf("};\n\n");
      }
   }
}

} // end namespace PV
