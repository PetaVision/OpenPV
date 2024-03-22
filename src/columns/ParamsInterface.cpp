/*
 * ParamsInterface.cpp
 *
 *  Created on May 16, 2018
 *      Author: Pete Schultz
 */

#include "ParamsInterface.hpp"

namespace PV {
ParamsInterface::~ParamsInterface() { free(mName); }

int ParamsInterface::initialize(char const *name, PVParams *params) {
   setName(name);
   setParams(params);
   setObjectType();
   setDescription(getObjectType() + " \"" + getName() + "\"");
   CheckpointerDataInterface::initialize();
   readParams();
   return PV_SUCCESS;
}

void ParamsInterface::setName(char const *name) {
   pvAssert(mName == nullptr);
   mName = strdup(name);
   FatalIf(mName == nullptr, "could not set name \"%s\". %s\n", name, strerror(errno));
}

void ParamsInterface::setParams(PVParams *params) { mParams = params; }

void ParamsInterface::setObjectType() {
   mObjectType = parameters()->groupKeywordFromName(getName());
}

void ParamsInterface::ioParams(enum ParamsIOFlag ioFlag, bool printHeader, bool printFooter) {
   if (printHeader) {
      ioParamsStartGroup(ioFlag);
   }
   ioParam_initializeFromCheckpointFlag(ioFlag);
   ioParamsFillGroup(ioFlag);
   if (printFooter) {
      ioParamsFinishGroup(ioFlag);
   }
}
/**
 * @brief initializeFromCheckpointFlag: If set to true, initialize using checkpoint direcgtory
 * set in HyPerCol.
 * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
 */
void ParamsInterface::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ or mWriteInitializeFromCheckpointFlag) {
      parameters()->ioParamValue(
            ioFlag,
            mName,
            "initializeFromCheckpointFlag",
            &mInitializeFromCheckpointFlag,
            mInitializeFromCheckpointFlag /*default value*/,
            false /*no warnings if param is absent*/);
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
