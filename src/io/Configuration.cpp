#include "Configuration.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

namespace PV {

Configuration::Configuration() {
   registerBooleanArgument("RequireReturn");
   registerStringArgument("OutputPath");
   registerStringArgument("ParamsFile");
   registerStringArgument("LogFile");
   registerStringArgument("GPUDevices");
   registerUnsignedIntArgument("RandomSeed");
   registerStringArgument("WorkingDirectory");
   registerBooleanArgument("Restart");
   registerStringArgument("CheckpointReadDirectory");
   registerIntOptionalArgument("NumThreads");
   registerIntegerArgument("NumRows");
   registerIntegerArgument("NumColumns");
   registerIntegerArgument("BatchWidth");
   registerBooleanArgument("DryRun");
}

void Configuration::registerArgument(std::string const &name, ConfigurationType type) {
   auto insertion = mConfigTypeMap.insert(std::make_pair(name, type));
   FatalIf(!insertion.second, "failed to register configuration argument %s\n", name.c_str());
   mConfigArguments.push_back(name);
}

void Configuration::registerBooleanArgument(std::string const &name) {
   registerArgument(name, CONFIG_BOOL);
   bool defaultValue = false;
   auto insertion    = mBooleanConfigMap.insert(std::make_pair(name, defaultValue));
   FatalIf(
         !insertion.second, "failed to register boolean configuration argument %s\n", name.c_str());
}

void Configuration::registerIntegerArgument(std::string const &name) {
   registerArgument(name, CONFIG_INT);
   int defaultValue = 0;
   auto insertion   = mIntegerConfigMap.insert(std::make_pair(name, defaultValue));
   FatalIf(
         !insertion.second, "failed to register integer configuration argument %s\n", name.c_str());
}

void Configuration::registerUnsignedIntArgument(std::string const &name) {
   registerArgument(name, CONFIG_UNSIGNED);
   unsigned int defaultValue = 0U;
   auto insertion            = mUnsignedIntConfigMap.insert(std::make_pair(name, defaultValue));
   FatalIf(
         !insertion.second,
         "failed to register unsigned int configuration argument %s\n",
         name.c_str());
}

void Configuration::registerStringArgument(std::string const &name) {
   registerArgument(name, CONFIG_STRING);
   std::string defaultValue{};
   auto insertion = mStringConfigMap.insert(std::make_pair(name, defaultValue));
   FatalIf(
         !insertion.second, "failed to register string configuration argument %s\n", name.c_str());
}

void Configuration::registerIntOptionalArgument(std::string const &name) {
   registerArgument(name, CONFIG_INT_OPTIONAL);
   IntOptional defaultValue;
   auto insertion = mIntOptionalConfigMap.insert(std::make_pair(name, defaultValue));
   FatalIf(
         !insertion.second,
         "failed to register optional int configuration argument %s\n",
         name.c_str());
}

Configuration::ConfigurationType Configuration::getType(std::string const &name) const {
   auto location = mConfigTypeMap.find(name);
   return (location == mConfigTypeMap.end()) ? CONFIG_UNRECOGNIZED : location->second;
}

bool const &Configuration::getBooleanArgument(std::string const &name) const {
   if (getType(name) != CONFIG_BOOL) {
      throw std::invalid_argument("getBooleanArgument");
   }
   auto location = mBooleanConfigMap.find(name);
   pvAssert(location != mBooleanConfigMap.end());
   return location->second;
}

int const &Configuration::getIntegerArgument(std::string const &name) const {
   if (getType(name) != CONFIG_INT) {
      throw std::invalid_argument("getIntegerArgument");
   }
   auto location = mIntegerConfigMap.find(name);
   pvAssert(location != mIntegerConfigMap.end());
   return location->second;
}

unsigned int const &Configuration::getUnsignedIntArgument(std::string const &name) const {
   if (getType(name) != CONFIG_UNSIGNED) {
      throw std::invalid_argument("getUnsignedIntArgument");
   }
   auto location = mUnsignedIntConfigMap.find(name);
   pvAssert(location != mUnsignedIntConfigMap.end());
   return location->second;
}

std::string const &Configuration::getStringArgument(std::string const &name) const {
   if (getType(name) != CONFIG_STRING) {
      throw std::invalid_argument("getStringArgument");
   }
   auto location = mStringConfigMap.find(name);
   pvAssert(location != mStringConfigMap.end());
   return location->second;
}

Configuration::IntOptional const &
Configuration::getIntOptionalArgument(std::string const &name) const {
   if (getType(name) != CONFIG_INT_OPTIONAL) {
      throw std::invalid_argument("getIntOptionalArgument");
   }
   auto location = mIntOptionalConfigMap.find(name);
   pvAssert(location != mIntOptionalConfigMap.end());
   return location->second;
}

std::string Configuration::printBooleanArgument(std::string const &name, bool const &value) {
   std::string returnedString(name);
   returnedString.append(":").append(value ? "true" : "false");
   return returnedString;
}

std::string Configuration::printIntegerArgument(std::string const &name, int const &value) {
   std::string returnedString(name);
   returnedString.append(":").append(std::to_string(value));
   return returnedString;
}

std::string
Configuration::printUnsignedArgument(std::string const &name, unsigned int const &value) {
   std::string returnedString(name);
   returnedString.append(":").append(std::to_string(value));
   return returnedString;
}

std::string Configuration::printStringArgument(std::string const &name, std::string const &value) {
   std::string returnedString(name);
   returnedString.append(":").append(value);
   return returnedString;
}

std::string
Configuration::printIntOptionalArgument(std::string const &name, IntOptional const &value) {
   std::string returnedString(name);
   returnedString.append(":");
   if (value.mUseDefault) {
      returnedString.append("-");
   }
   else {
      returnedString.append(std::to_string(value.mValue));
   }
   return returnedString;
}

std::string Configuration::printArgument(std::string const &name) const {
   ConfigurationType type = getType(name);
   std::string returnString;
   bool status; // Used to verify the result of calling the get*Argument method.
   switch (type) {
      case CONFIG_UNRECOGNIZED: break;
      case CONFIG_BOOL: returnString = printBooleanArgument(name, getBooleanArgument(name)); break;
      case CONFIG_INT: returnString  = printIntegerArgument(name, getIntegerArgument(name)); break;
      case CONFIG_UNSIGNED:
         returnString = printUnsignedArgument(name, getUnsignedIntArgument(name));
         break;
      case CONFIG_STRING: returnString = printStringArgument(name, getStringArgument(name)); break;
      case CONFIG_INT_OPTIONAL:
         returnString = printIntOptionalArgument(name, getIntOptionalArgument(name));
         break;
      default: pvAssert(0); break;
   }
   return returnString;
}

std::string Configuration::printConfig() const {
   std::string configString;
   for (auto &s : mConfigArguments) {
      configString.append(printArgument(s)).append("\n");
   }
   return configString;
}

bool Configuration::setArgumentUsingString(std::string const &name, std::string const &value) {
   bool found;
   switch (getType(name)) {
      case CONFIG_UNRECOGNIZED: return false; break;
      case CONFIG_BOOL: setBooleanArgument(name, parseBoolean(value)); break;
      case CONFIG_INT: setIntegerArgument(name, parseInteger(value)); break;
      case CONFIG_UNSIGNED: setUnsignedIntArgument(name, parseUnsignedInt(value)); break;
      case CONFIG_STRING: setStringArgument(name, parseString(value)); break;
      case CONFIG_INT_OPTIONAL: setIntOptionalArgument(name, parseIntOptional(value)); break;
      default: pvAssert(0);
   }
   return true;
}

bool Configuration::parseBoolean(std::string const &valueString) const {
   bool value;
   if (valueString == "true" || valueString == "T" || valueString == "1") {
      value = true;
   }
   else if (valueString == "false" || valueString == "F" || valueString == "0") {
      value = false;
   }
   else {
      throw std::invalid_argument("parseBoolean");
   }
   return value;
}

int Configuration::parseInteger(std::string const &valueString) const {
   int value = std::stoi(valueString);
   return value;
}

unsigned int Configuration::parseUnsignedInt(std::string const &valueString) const {
   unsigned int value = (unsigned int)std::stoul(valueString);
   return value;
}

std::string Configuration::parseString(std::string const &valueString) const { return valueString; }

Configuration::IntOptional Configuration::parseIntOptional(std::string const &valueString) const {
   IntOptional value;
   if (valueString.empty() || valueString == "-") {
      value.mUseDefault = true;
      value.mValue      = -1;
   }
   else {
      value.mUseDefault = false;
      value.mValue      = std::stoi(valueString);
   }
   return value;
}

bool Configuration::setBooleanArgument(std::string const &name, bool const &value) {
   bool found = getType(name) == CONFIG_BOOL;
   if (found) {
      auto location = mBooleanConfigMap.find(name);
      pvAssert(location != mBooleanConfigMap.end());
      location->second = value;
   }
   return found;
}

bool Configuration::setIntegerArgument(std::string const &name, int const &value) {
   bool found = getType(name) == CONFIG_INT;
   if (found) {
      auto location = mIntegerConfigMap.find(name);
      pvAssert(location != mIntegerConfigMap.end());
      location->second = value;
   }
   return found;
}

bool Configuration::setUnsignedIntArgument(std::string const &name, unsigned int const &value) {
   bool found = getType(name) == CONFIG_UNSIGNED;
   if (found) {
      auto location = mUnsignedIntConfigMap.find(name);
      pvAssert(location != mUnsignedIntConfigMap.end());
      location->second = value;
   }
   return found;
}

bool Configuration::setStringArgument(std::string const &name, std::string const &value) {
   bool found = getType(name) == CONFIG_STRING;
   if (found) {
      auto location = mStringConfigMap.find(name);
      pvAssert(location != mStringConfigMap.end());
      location->second = value;
   }
   return found;
}

bool Configuration::setIntOptionalArgument(std::string const &name, IntOptional const &value) {
   bool found = getType(name) == CONFIG_INT_OPTIONAL;
   if (found) {
      auto location = mIntOptionalConfigMap.find(name);
      pvAssert(location != mIntOptionalConfigMap.end());
      location->second = value;
   }
   return found;
}

} // end namespace PV
