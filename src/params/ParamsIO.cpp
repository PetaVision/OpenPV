#include "ParamsIO.hpp"
#include <climits>
#include <cmath>

namespace PV {

ParamsIO::ParamsIO(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mParams   = params;
   mDefaults = defaults;

   mPrintParamsStream = nullptr;
   mPrintLuaStream    = nullptr;
}

Parameter::Type
ParamsIO::checkType(std::string const &paramName) const {
   std::string const &keyword = mParams->getKeyword();
   Parameter::Type result = mParams->checkType(paramName);
   if (result == Parameter::Type::NotFound) {
      if (mDefaults != nullptr) {
         result = mDefaults->checkType(paramName);
      }
   }
   return result;
}

template<>
int ParamsIO::convertDoubleToArithmeticType<int>(double value) {
   int y = 0;
   if (value >= static_cast<double>(INT_MAX)) {
      y = INT_MAX;
   }
   else if (value <= static_cast<double>(INT_MIN)) {
      y = INT_MIN;
   }
   else {
      y = static_cast<int>(std::nearbyint(value));
   }
   return y;
}

void ParamsIO::handleUnnecessaryParameter(std::string const &paramName) {
   int status = PV_SUCCESS;
   if (mParams->present(paramName)) {
      WarnLog().printf(
            "%s \"%s\" does not use parameter %s, but it is present in the parameters file.\n",
            getKeyword().c_str(),
            getName().c_str(),
            paramName.c_str());
      // mark param as read so that presentAndNotBeenRead doesn't trip up
      auto type = checkType(paramName);
      switch (type) {
         case Parameter::Type::NotFound:
            Fatal().printf(
                  "handleUnnecessaryParameter(\"%s\", \"%s\"): "
                  "isPresent() returned true but checkType() was NotFound\n",
                  getKeyword().c_str(), getName().c_str());
            break;
         case Parameter::Type::Numeric: mParams->read<double>(paramName); break;
         case Parameter::Type::Array:   mParams->read<std::vector<double>>(paramName); break;
         case Parameter::Type::String:  mParams->read<std::string>(paramName); break;
         default:
            Fatal().printf(
                  "handleUnnecessaryParameter(\"%s\", \"%s\"): "
                  "checkType() returned unrecognized type %d.\n",
                  getKeyword().c_str(), getName().c_str(), static_cast<int>(type));

      }
   }
}

void ParamsIO::handleUnnecessaryCaseInsensitiveParameter(
      std::string const &param_name, std::string const &correct_value) {
   int status             = PV_SUCCESS;
   if (isPresent(param_name)) {
      WarnLog().printf(
            "%s \"%s\" does not use string parameter %s, but it is present in the parameters "
            "file.\n",
            getKeyword(),
            getName(),
            param_name);
      std::string const &params_value = readString(param_name, false /*warnIfAbsentFlag*/);
      // marks param as read so that presentAndNotBeenRead doesn't trip up

      // Check against correct value.
      std::string correct_value_i(correct_value);
      std::string params_value_i(params_value);
      for (char &c : params_value_i) {
         c = (char)tolower((int)c);
      }
      for (char &c : correct_value_i) {
         c = (char)tolower((int)c);
      }
      FatalIf(
            params_value_i != correct_value_i,
            "%s \"%s\": parameter string %s = \"%s\" is inconsistent with correct value \"%s\". "
            "Exiting.\n",
            getKeyword(),
            getName(),
            param_name,
            params_value.c_str(),
            correct_value);
   }
}

bool ParamsIO::hasBeenRead(char const *param_name) {
   return mParams->hasBeenRead(param_name);
}

bool ParamsIO::isPresent(std::string const &paramName) {
   // Check Params but do not check Defaults
   // (isArray(), isNumeric(), isString() all check both Params and Defaults)
   return mParams->checkType(paramName) != Parameter::Type::NotFound;
}

bool ParamsIO::isArray(std::string const &paramName) const {
   auto type = checkType(paramName);
   return type == Parameter::Type::Array or type == Parameter::Type::Numeric;
}

bool ParamsIO::isNumeric(std::string const &paramName) const {
   return checkType(paramName) == Parameter::Type::Numeric;
}

bool ParamsIO::isString(std::string const &paramName) const {
   return checkType(paramName) == Parameter::Type::String;
}

bool ParamsIO::presentAndNotBeenRead(char const *param_name) {
   bool is_present = isPresent(param_name);
   bool has_been_read = hasBeenRead(param_name);
   return is_present && !has_been_read;
}

double ParamsIO::readDouble(std::string const &paramName, bool warnIfAbsentFlag) {
   assert(mParams != nullptr);
   std::string const &groupName = mParams->getName();
   std::string const &keyword = mParams->getKeyword();
   double const *valuePtr = nullptr;
   Parameter::Type paramType = mParams->checkType(paramName);
   if (paramType == Parameter::Type::Numeric) {
      valuePtr = mParams->read<double>(paramName);
      assert(valuePtr != nullptr);
   }
   else if (paramType == Parameter::Type::NotFound) {
      if (mDefaults != nullptr) {
         valuePtr = mDefaults->peek<double>(paramName);
         if (valuePtr != nullptr and warnIfAbsentFlag) {
            WarnLog().printf(
                  "Using default value %f for parameter \"%s\" in group \"%s\"\n",
                  *valuePtr, paramName.c_str(), groupName.c_str());
         }
      }
      else {
         ErrorLog().printf("No default params for %s were set.\n", keyword.c_str());
      }
   }
   else {
      ErrorLog().printf("Parameter %s in group %s exists but is non-numeric.\n",
            paramName.c_str(), groupName.c_str());
   }
   FatalIf(
         valuePtr == nullptr,
         "Numeric parameter \"%s\" was not defined in %s \"%s\" and no default value was found.\n",
         paramName.c_str(), keyword.c_str(), groupName.c_str());
   return *valuePtr;
}

std::string const &ParamsIO::readString(std::string const &paramName, bool warnIfAbsentFlag) {
   assert(mParams != nullptr);
   std::string const &groupName = mParams->getName();
   std::string const &keyword = mParams->getKeyword();
   std::string const *valuePtr = nullptr;
   Parameter::Type paramType = mParams->checkType(paramName);
   if (paramType == Parameter::Type::String) {
      valuePtr = mParams->read<std::string>(paramName);
      assert(valuePtr != nullptr);
   }
   else if (paramType == Parameter::Type::NotFound) {
      if (mDefaults != nullptr) {
         paramType = mDefaults->checkType(paramName);
         if (paramType == Parameter::Type::String) {
            valuePtr = mDefaults->peek<std::string>(paramName);
            if (warnIfAbsentFlag) {
               WarnLog().printf(
                     "Using default value \"%s\" for parameter \"%s\" in group \"%s\"\n",
                     valuePtr->c_str(),
                     paramName.c_str(),
                     groupName.c_str());
            }
         }
         else if (paramType == Parameter::Type::NotFound) {
            ErrorLog().printf(
                  "No default value for %s param \"%s\".\n",
                  keyword.c_str(), paramName.c_str());
         }
         else {
            ErrorLog().printf(
                  "Default value for %s param \"%s\" is not a string.\n",
                  keyword.c_str(), paramName.c_str());
         }
      }
      else {
         ErrorLog().printf("No default params for %s were set.\n", keyword.c_str());
      }
   }
   else {
      ErrorLog().printf("Parameter %s in group %s exists but is not a string.\n",
            paramName.c_str(), groupName.c_str());
   }
   FatalIf(
         valuePtr == nullptr,
         "String parameter \"%s\" was not defined in %s \"%s\" and no default value was found.\n",
         paramName.c_str(), keyword.c_str(), groupName.c_str());
   return *valuePtr;
}

std::vector<double> const *ParamsIO::readArray(std::string const &paramName, bool warnIfAbsentFlag) {
   assert(mParams != nullptr);
   std::string const &groupName = mParams->getName();
   std::string const &keyword = mParams->getKeyword();
   std::vector<double> const *valuePtr = mParams->read<std::vector<double>>(paramName);
   if (valuePtr == nullptr) {
      if (mDefaults != nullptr) {
         valuePtr = mDefaults->peek<std::vector<double>>(paramName);
         if (valuePtr != nullptr and warnIfAbsentFlag) {
            WarnLog().printf(
                  "Using default value %s for parameter \"%s\" in group \"%s\"\n",
                  paramToString(*valuePtr).c_str(),
                  paramName.c_str(),
                  groupName.c_str());
         }
      }
      else {
         ErrorLog().printf("No default params for %s were set.\n", keyword.c_str());
      }
   }
   FatalIf(
         valuePtr == nullptr,
         "Array parameter \"%s\" was not defined in %s \"%s\" and no default value was found.\n",
         paramName.c_str(), keyword.c_str(), groupName.c_str());
   return valuePtr;
}

template <>
std::string ParamsIO::paramToString<bool>(bool const &paramValue) {
   return std::string(paramValue ? "true" : "false");
}

} // end namespace PV
