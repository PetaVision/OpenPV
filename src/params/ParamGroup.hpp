#ifndef PARAMGROUP_HPP_
#define PARAMGROUP_HPP_

#include "params/Parameter.hpp"

#include <map>
#include <string>
#include <vector>

namespace PV {

class ParamGroup {
  public:
   typedef std::map<std::string, Parameter>::value_type value_type;
   typedef std::map<std::string, Parameter>::reference reference;
   typedef std::map<std::string, Parameter>::const_reference const_reference;
   typedef std::map<std::string, Parameter>::iterator iterator;
   typedef std::map<std::string, Parameter>::const_iterator const_iterator;
   typedef std::map<std::string, Parameter>::difference_type difference_type;
   typedef std::map<std::string, Parameter>::size_type size_type;

   ParamGroup(std::string const &name, std::string const &keyword, int processRank);

   Parameter::Type checkType(std::string const &paramName) const;
   void clearAllHasBeenReadFlags();
   void clearHasBeenReadFlag(std::string const &paramName);
   bool erase(std::string const &paramName);
   bool hasBeenRead(std::string const &paramName);

   template <typename T>
   bool insert(std::string const &paramName, T const &value);

   bool isArray(std::string const &paramName) const;
   bool isNumeric(std::string const &paramName) const;
   bool isString(std::string const &paramName) const;

   /**
    * lookForUnread() tests each parameter in the parameter group for whether it's been read.
    * It returns a vector of strings, each string the name of one unread parameter.
    */
   std::vector<std::string> lookForUnread();

   template <typename T>
   T const *peek(std::string const &paramName) const; 

   bool present(std::string const &paramName);

   template <typename T>
   T const *read(std::string const &paramName); 

   template <typename T>
   bool replace(std::string const &paramName, T const &value);

   iterator begin() { return mParameterMap.begin(); }
   const_iterator begin() const { return mParameterMap.begin(); }
   const_iterator cbegin() const { return mParameterMap.begin(); }
   iterator end() { return mParameterMap.end(); }
   const_iterator end() const { return mParameterMap.end(); }
   const_iterator cend() const { return mParameterMap.end(); }
   bool operator==(ParamGroup const &rhs) const;
   bool operator!=(ParamGroup const &rhs) const { return !(*this == rhs); }
   void swap(ParamGroup &rhs);
   size_type size() const { return mParameterMap.size(); }
   size_type max_size() const { return mParameterMap.max_size(); }
   bool empty() const { return mParameterMap.empty(); }

   std::string const &getKeyword() const { return mKeyword; }
   std::string const &getName() const { return mName; }

  private:
   std::string mKeyword;
   std::string mName;
   std::map<std::string, Parameter> mParameterMap;
   int mProcessRank;
   static const Parameter::Type mNotFound = Parameter::Type::NotFound;
};

template <typename T>
bool ParamGroup::insert(std::string const &paramName, T const &paramValue) {
   auto insertResult = mParameterMap.emplace(paramName, paramValue);
   return insertResult.second;
}

template <typename T>
T const *ParamGroup::peek(std::string const &paramName) const {
   auto findResult = mParameterMap.find(paramName); 
   if (findResult != mParameterMap.end()) {
      return findResult->second.peek<T>();
   }
   else {
      return nullptr;
   }
}

template <typename T>
T const *ParamGroup::read(std::string const &paramName) {
   auto findResult = mParameterMap.find(paramName); 
   if (findResult != mParameterMap.end()) {
      return findResult->second.read<T>();
   }
   else {
      return nullptr;
   }
}

template <typename T>
bool ParamGroup::replace(std::string const &paramName, T const &value) {
    bool isPresent = present(paramName);
    if (!isPresent) { return false; }
    erase(paramName);
    return insert<T>(paramName, value);
}

void swap(ParamGroup &lhs, ParamGroup &rhs);

} // namespace PV

namespace std {

template <>
void swap<PV::ParamGroup>(PV::ParamGroup &lhs, PV::ParamGroup &rhs);

} // namespace std

#endif // PARAMGROUP_HPP_
