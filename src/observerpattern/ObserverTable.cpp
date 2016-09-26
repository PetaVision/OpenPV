/*
 * ObserverTable.cpp
 *
 *  Created on: Jul 22, 2016
 *      Author: pschultz
 */

#include "observerpattern/ObserverTable.hpp"
#include "utils/PVAssert.hpp"
#include <algorithm>

namespace PV {

std::vector<Observer*>::size_type ObserverTable::size() const {
   pvAssert(mObjectVector.size()==mObjectMap.size());
   return mObjectVector.size();
}

bool ObserverTable::addObject(std::string const& name, Observer * entry) {
   bool addSucceeded = mObjectMap.insert(std::make_pair(std::string(name), entry)).second; // map::insert() returns a pair whose second element is whether the insertion was successful.
   if (addSucceeded) {
      mObjectVector.emplace_back(entry);
   }
   return addSucceeded;
}

void ObserverTable::deleteObject(std::string const& name, bool deallocateFlag) {
   Observer * obj = nullptr;
   auto mapSearchResult = mObjectMap.find(name);
   if (mapSearchResult==mObjectMap.end()) {
      obj = mapSearchResult->second;
      auto vectorSearchResult = find(mObjectVector.begin(), mObjectVector.end(), obj);
      pvAssert(vectorSearchResult!=mObjectVector.end());
      mObjectMap.erase(mapSearchResult);
      mObjectVector.erase(vectorSearchResult);
      if (deallocateFlag) { delete obj; }
   }
}

void ObserverTable::clear(bool deallocateFlag) {
   if (deallocateFlag) {
      for (auto& obj : mObjectVector) {
         delete obj;
      }
   }
   mObjectVector.clear();
   mObjectMap.clear();
}

} /* namespace PV */
