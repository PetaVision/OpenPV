/*
 * ParamsInterface.cpp
 *
 *  Created on May 16, 2018
 *      Author: Pete Schultz
 */

#include "ParamsInterface.hpp"

namespace PV {
ParamsInterface::~ParamsInterface() {}

int ParamsInterface::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   FatalIf(paramsIO == nullptr, "ParamsInterface called with null ParamsIO\n");
   setName(paramsIO->getName());
   setKeyword(paramsIO->getKeyword());
   setParams(paramsIO);
   setObjectType();
   setDescription(getObjectType() + " \"" + getName() + "\"");
   CheckpointerDataInterface::initialize();
   ioParams(ParamsIOSwitch::Read, false, false);
   return PV_SUCCESS;
}

void ParamsInterface::setName(std::string const &name) {
   pvAssert(mName.empty());
   mName = name;
}

void ParamsInterface::setKeyword(std::string const &keyword) {
   pvAssert(mKeyword.empty());
   mKeyword = keyword;
}

void ParamsInterface::setParams(std::shared_ptr<ParamsIO> paramsIO) {
   mParamsIO = paramsIO;
}

void ParamsInterface::setObjectType() {
   mObjectType = getKeyword();
}

void ParamsInterface::ioParams(ParamsIOSwitch ioSwitch, bool printHeader, bool printFooter) {
   if (printHeader) {
      ioParamsStartGroup(ioSwitch);
   }
   ioParam_initializeFromCheckpointFlag(ioSwitch);
   ioParamsFillGroup(ioSwitch);
   if (printFooter) {
      ioParamsFinishGroup(ioSwitch);
   }
}
/**
 * @brief initializeFromCheckpointFlag: If set to true, initialize using checkpoint directory
 * set in HyPerCol.
 * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
 */
void ParamsInterface::ioParam_initializeFromCheckpointFlag(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      if (mParamsIO->isNumeric("initializeFromCheckpointFlag")) {
         mParamsIO->ioParam(
               ioSwitch,
               "initializeFromCheckpointFlag",
               &mInitializeFromCheckpointFlag,
               false /*warnIfAbsentFlag*/);
      }
   }
   else {
      pvAssert(ioSwitch == ParamsIOSwitch::Write);
      if (mWriteInitializeFromCheckpointFlag) {
         mParamsIO->ioParam(
               ioSwitch,
               "initializeFromCheckpointFlag",
               &mInitializeFromCheckpointFlag,
               false /*warnIfAbsentFlag*/);
      }
   }
}

void ParamsInterface::ioParamsStartGroup(ParamsIOSwitch ioSwitch) {
   auto *printParamsStream = mParamsIO->getPrintParamsStream();
   if (printParamsStream) {
      printParamsStream->printf("\n");
      printParamsStream->printf("%s \"%s\" = {\n", getKeyword(), getName());
   }
   auto *printLuaStream = mParamsIO->getPrintLuaStream();
   if (printLuaStream) {
      printLuaStream->printf("\n");
      printLuaStream->printf("%s \"%s\" = {\n", getKeyword(), getName());
   }
}

void ParamsInterface::ioParamsFinishGroup(ParamsIOSwitch ioSwitch) {
   auto *printParamsStream = mParamsIO->getPrintParamsStream();
   if (printParamsStream) {
         printParamsStream->printf("};\n");
   }
   auto *printLuaStream = mParamsIO->getPrintLuaStream();
   if (printLuaStream) {
         printLuaStream->printf("};\n\n");
   }
}

} // end namespace PV
