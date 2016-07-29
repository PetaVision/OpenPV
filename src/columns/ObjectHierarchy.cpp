/*
 * ObjectHierarchy.cpp
 *
 *  Created on: Jul 22, 2016
 *      Author: pschultz
 */

#include <columns/ObjectHierarchy.hpp>
#include <algorithm>

namespace PV {

ObjectHierarchy::ObjectHierarchy() {
}

ObjectHierarchy::~ObjectHierarchy() {
}

bool ObjectHierarchy::addObject(BaseObject * obj) {
   bool addSucceeded = mObjectMap.insert(std::make_pair(obj->getName(), obj)).second;
   if (addSucceeded) {
      mObjectVector.emplace_back(obj);
   }
   return addSucceeded;
}

void ObjectHierarchy::deleteObject(std::string const& name, bool deallocateFlag) {
   BaseObject * obj = nullptr;
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

void ObjectHierarchy::clear(bool deallocateFlag) {
   if (deallocateFlag) {
      for (auto& obj : mObjectVector) {
         delete obj;
      }
   }
   mObjectVector.clear();
   mObjectMap.clear();
}

} /* namespace PV */
