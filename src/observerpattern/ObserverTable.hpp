/*
 * ObserverTable.hpp
 *
 *  Created on: Jul 22, 2016
 *      Author: pschultz
 */

#ifndef OBSERVERTABLE_HPP_
#define OBSERVERTABLE_HPP_

#include "observerpattern/Observer.hpp"
#include <map>
#include <string>
#include <vector>

namespace PV {

/**
 * An encapsulation of a map of name-object pairs and a vector of objects.
 * The map has the object names as the keys, and pointers to objects as the
 * values.
 * The vector has the same set of objects as the map's values.
 * By keeping the vector, we can guarantee the order in which we iterate through
 * the objects.
 * By keeping the map, we have an easy way to look up the object from the name.
 */
class ObserverTable {
  public:
   ObserverTable() {}
   virtual ~ObserverTable() {}

   std::vector<Observer *> const &getObjectVector() const { return mObjectVector; }
   std::map<std::string, Observer *> const &getObjectMap() const { return mObjectMap; }
   Observer *getObject(std::string const &name) const {
      auto lookupResult = mObjectMap.find(name);
      return lookupResult == mObjectMap.end() ? nullptr : lookupResult->second;
   }
   Observer *getObject(char *name) const { return getObject(std::string(name)); };
   std::vector<Observer *>::size_type size() const;
   bool addObject(std::string const &name, Observer *entry);
   void deleteObject(std::string const &name, bool deallocateFlag);
   void deleteObject(char const *name, bool deallocateFlag) {
      deleteObject(std::string(name), deallocateFlag);
   }
   void clear(bool deallocateFlag);

   template <typename S>
   S *lookup(std::string const &name) const {
      return lookup<S>(name.c_str());
   }

   template <typename S>
   S *lookup(char const *name) const {
      S *lookupResult = nullptr;
      auto findResult = mObjectMap.find(name);
      if (findResult != mObjectMap.end()) {
         auto observerPtr = findResult->second;
         lookupResult     = dynamic_cast<S *>(observerPtr);
      }
      return lookupResult;
   }

   // To iterate over ObserverTable:
   typedef std::vector<Observer *>::iterator iterator;
   typedef std::vector<Observer *>::const_iterator const_iterator;
   typedef std::vector<Observer *>::reverse_iterator reverse_iterator;
   typedef std::vector<Observer *>::const_reverse_iterator const_reverse_iterator;

   iterator begin() { return mObjectVector.begin(); }
   const_iterator begin() const { return mObjectVector.begin(); }
   const_iterator cbegin() const { return mObjectVector.cbegin(); }
   iterator end() { return mObjectVector.end(); }
   const_iterator end() const { return mObjectVector.end(); }
   const_iterator cend() const { return mObjectVector.cend(); }

  private:
   std::vector<Observer *> mObjectVector;
   std::map<std::string, Observer *> mObjectMap;
};

} /* namespace PV */

#endif /* OBSERVERTABLE_HPP_ */
