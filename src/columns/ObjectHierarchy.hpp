/*
 * ObjectHierarchy.hpp
 *
 *  Created on: Jul 22, 2016
 *      Author: pschultz
 */

#ifndef OBJECTHIERARCHY_HPP_
#define OBJECTHIERARCHY_HPP_

#include "columns/BaseObject.hpp"
#include <vector>
#include <map>

namespace PV {

/**
 * An encapsulation of a map of name-object pairs and a vector of objects.
 * The map has the object names as the keys, and pointers to objects as the values.
 * The vector has the same set of objects as the map's values.
 * By keeping the vector, we can guarantee the order in which we iterate through the objects.
 * By keeping the map, we have an easy way to look up the object from the name.
 */
class ObjectHierarchy {
public:
   ObjectHierarchy();
   virtual ~ObjectHierarchy();

   std::vector<BaseObject*> const& getObjectVector() const { return mObjectVector; }
   std::map<std::string, BaseObject*> const& getObjectMap() const { return mObjectMap; }
   BaseObject * getObject(std::string const& name) const {
      auto lookupResult = mObjectMap.find(name);
      return lookupResult==mObjectMap.end() ? nullptr : lookupResult->second;
   }
   BaseObject * getObject(char * name) const {
      return getObject(std::string(name));
   };
   std::vector<BaseObject*>::size_type size() const {
      pvAssert(mObjectVector.size()==mObjectMap.size());
      return mObjectVector.size();
   }
   bool addObject(BaseObject *);
   void deleteObject(std::string const& name, bool deallocateFlag);
   void deleteObject(char const * name, bool deallocateFlag) { deleteObject(std::string(name), deallocateFlag); }
   void clear(bool deallocateFlag);

private:
   std::vector<BaseObject*> mObjectVector;
   std::map<std::string, BaseObject*> mObjectMap;
};

} /* namespace PV */

#endif /* OBJECTHIERARCHY_HPP_ */
