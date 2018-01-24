/*
 * ObjectMapComponent.hpp
 *
 *  Created on: Nov 20, 2017
 *      Author: pschultz
 */

#ifndef OBJECTMAPCOMPONENT
#define OBJECTMAPCOMPONENT

#include "columns/BaseObject.hpp"
#include <map>

namespace PV {

/**
 * A BaseObject containing an object map as its main data member
 * The motivation is to provide a way for components of a HyPerCol to pass
 * the HyPerCol's table of components to its subcomponents during the
 * CommunicateInitInfo phase.
 */
class ObjectMapComponent : public BaseObject {
  public:
   ObjectMapComponent(char const *name, HyPerCol *hc) { initialize(name, hc); }

   virtual ~ObjectMapComponent() {}

   virtual void setObjectType() override { mObjectType = "ObjectMapComponent"; }

   void setObjectMap(std::map<std::string, Observer *> const &table) { mObjectMap = table; }

   template <typename S>
   S *lookup(std::string const &name) const {
      S *lookupResult = nullptr;
      auto findResult = mObjectMap.find(name);
      if (findResult != mObjectMap.end()) {
         auto observerPtr = findResult->second;
         lookupResult     = dynamic_cast<S *>(observerPtr);
      }
      return lookupResult;
   }

  protected:
   ObjectMapComponent() {}

   int initialize(char const *name, HyPerCol *hc) { return BaseObject::initialize(name, hc); }

  protected:
   std::map<std::string, Observer *> mObjectMap;

}; // class ObjectMapComponent

} // namespace PV

#endif // OBJECTMAPCOMPONENT
