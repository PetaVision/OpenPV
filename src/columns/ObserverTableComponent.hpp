/*
 * ObserverTableComponent.hpp
 *
 *  Created on: Nov 20, 2017
 *      Author: pschultz
 */

#ifndef OBSERVERTABLECOMPONENT_HPP_
#define OBSERVERTABLECOMPONENT_HPP_

#include "columns/BaseObject.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

/**
 * A BaseObject containing an object map as its main data member
 * The motivation is to provide a way for components of a HyPerCol to pass
 * the HyPerCol's table of components to its subcomponents during the
 * CommunicateInitInfo phase.
 */
class ObserverTableComponent : public BaseObject {
  public:
   ObserverTableComponent(char const *name, HyPerCol *hc) { initialize(name, hc); }

   virtual ~ObserverTableComponent() {}

   virtual void setObjectType() override { mObjectType = "ObserverTableComponent"; }

   void setObserverTable(ObserverTable const &table) { mObserverTable = table; }

   ObserverTable const &getObserverTable() const { return mObserverTable; }

  protected:
   ObserverTableComponent() {}

   int initialize(char const *name, HyPerCol *hc) { return BaseObject::initialize(name, hc); }

  protected:
   ObserverTable mObserverTable;

}; // class ObserverTableComponent

} // namespace PV

#endif // OBSERVERTABLECOMPONENT_HPP_
