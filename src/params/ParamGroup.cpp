#include <algorithm>

#include "ParamGroup.hpp"
#include "utils/PVLog.hpp"

namespace PV {

ParamGroup::ParamGroup(std::string const &name, std::string const &keyword, int processRank) {
    mName = name;
    mKeyword = keyword;
    mProcessRank = processRank;
}

Parameter::Type ParamGroup::checkType(std::string const &paramName) const {
   auto findResult = mParameterMap.find(paramName);
   if (findResult == mParameterMap.end()) {
      return mNotFound;
   }
   else {
      return findResult->second.getType();
   }
}

void ParamGroup::clearAllHasBeenReadFlags() {
   for (auto &p : mParameterMap) {
      p.second.clearHasBeenReadFlag();
   }
}

void ParamGroup::clearHasBeenReadFlag(std::string const &paramName) {
   auto findResult = mParameterMap.find(paramName);
   if (findResult != mParameterMap.end()) {
      findResult->second.clearHasBeenReadFlag();
   }
}

bool ParamGroup::erase(std::string const &paramName) {
   auto eraseResult = mParameterMap.erase(paramName);
   return static_cast<bool>(eraseResult);
}

bool ParamGroup::hasBeenRead(std::string const &paramName) {
   auto findResult = mParameterMap.find(paramName);
   if (findResult != mParameterMap.end()) {
      return findResult->second.getHasBeenReadFlag();
   }
   else {
      return false;
   }
}

bool ParamGroup::isArray(std::string const &paramName) const {
   auto type = checkType(paramName);
   return type == Parameter::Type::Array or type == Parameter::Type::Numeric;
}

bool ParamGroup::isNumeric(std::string const &paramName) const {
   return checkType(paramName) == Parameter::Type::Numeric;
}

bool ParamGroup::isString(std::string const &paramName) const {
   return checkType(paramName) == Parameter::Type::String;
}


bool ParamGroup::lookForUnread(bool errorOnUnreadFlag) {
   bool status = false;
   for (auto const &p : mParameterMap) {
      std::string const &name = p.first;
      Parameter const &param = p.second;
      if (!param.getHasBeenReadFlag()) {
         if (mProcessRank == 0) {
            std::string message("Parameter group \"#1\": parameter \"#2\" has not been read.\n");
            message.replace(message.find("#1"), 2, mName);
            message.replace(message.find("#2"), 2, name);
            if (errorOnUnreadFlag) {
               ErrorLog() << message;
            }
            else {
               WarnLog() << message;
            }
         }
         status = true;
      }
   }
   return status;
}

bool ParamGroup::present(std::string const &paramName) {
   auto findResult = mParameterMap.find(paramName);
   bool isPresent = findResult != mParameterMap.end();
   return isPresent;
}

bool ParamGroup::operator==(ParamGroup const &rhs) const {
   if (rhs.size() != size()) { return false; }
   if (rhs.getKeyword() != getKeyword()) { return false; }
   return std::equal(begin(), end(), rhs.begin());
}

void ParamGroup::swap(ParamGroup &rhs) {
   std::swap(mKeyword, rhs.mKeyword);
   std::swap(mName, rhs.mName);
   std::swap(mParameterMap, rhs.mParameterMap);
   std::swap(mProcessRank, rhs.mProcessRank);
}

void swap(ParamGroup &lhs, ParamGroup &rhs) {
   lhs.swap(rhs);
}

} // namespace PV

namespace std {

template <>
void swap<PV::ParamGroup>(PV::ParamGroup &lhs, PV::ParamGroup &rhs) {
   lhs.swap(rhs);
}

} // namespace std
