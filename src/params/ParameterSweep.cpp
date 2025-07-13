#include "ParameterSweep.hpp"

#include "include/pv_common.h"
#include "utils/PVLog.hpp"

#include <cassert>

namespace PV {

ParameterSweep::ParameterSweep() {}

ParameterSweep::~ParameterSweep() {}

int ParameterSweep::setGroupAndParameter(
        std::string const &groupName, std::string const &paramName) {
   int status = PV_SUCCESS;
   if (!mGroupName.empty() or !mParamName.empty()) {
      ErrorLog(errorMessage);
      errorMessage.printf("ParameterSweep::setGroupParameter: ");
      if (!mGroupName.empty()) {
         errorMessage.printf(" group name has already been set to \"%s\".", mGroupName.c_str());
      }
      if (!mParamName.empty()) {
         errorMessage.printf(" param name has already been set to \"%s\".", mParamName.c_str());
      }
      errorMessage.printf("\n");
      status = PV_FAILURE;
   }
   else {
      mGroupName = groupName;
      mParamName = paramName;
      // Check for duplicates
   }
   return status;
}

int ParameterSweep::pushNumericValue(double val) {
   int status = PV_SUCCESS;
   if (mNumValues == 0) {
      assert(mValuesNumber.empty());
      mType = SWEEP_NUMBER;
   }
   else {
      FatalIf(
            mType != SWEEP_NUMBER,
            "Pushing numeric value to non-numeric parameter sweep %s \"%s\".n",
            mGroupName.c_str(), mParamName.c_str());
   }
   assert(mType == SWEEP_NUMBER);
   assert(mValuesString.empty());

   mValuesNumber.emplace_back(val);
   ++mNumValues;
   assert(mNumValues == static_cast<int>(mValuesNumber.size()));
   return status;
}

int ParameterSweep::pushStringValue(std::string const &sval) {
   int status = PV_SUCCESS;
   if (mNumValues == 0) {
      assert(mValuesString.empty());
      mType = SWEEP_STRING;
   }
   else {
      FatalIf(
            mType != SWEEP_STRING,
            "Pushing string value to non-string parameter sweep %s \"%s\".n",
            mGroupName.c_str(), mParamName.c_str());
   }
   assert(mType == SWEEP_STRING);
   assert(mValuesNumber.empty());

   mValuesString.emplace_back(sval);
   ++mNumValues;
   assert(mNumValues == static_cast<int>(mValuesString.size()));
   return status;
}

int ParameterSweep::getNumericValue(int n, double *val) {
   int status = PV_SUCCESS;
   if (mType != SWEEP_NUMBER or n < 0 or n >= mNumValues) {
      status = PV_FAILURE;
   }
   else {
      *val = mValuesNumber[n];
   }
   return status;
}

int ParameterSweep::getStringValue(int n, char const **sval) {
   int status = PV_SUCCESS;
   if (mType != SWEEP_STRING or n < 0 or n >= mNumValues) {
      status = PV_FAILURE;
   }
   else {
      *sval = mValuesString[n].c_str();
   }
   return status;
}

} // namespace PV
