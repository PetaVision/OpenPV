/*
 * ComponentBasedObject.hpp
 *
 *  This is the base class for layers, connections, and any
 *  other object that needs to inherit Subject (e.g. for an
 *  observer table for its own components) and BaseObject
 *  (e.g. to be part of the HyPerCol observer table).
 *
 *  Created on: Jun 11, 2016
 *      Author: pschultz
 */

#ifndef COMPONENTBASEDOBJECT_HPP_
#define COMPONENTBASEDOBJECT_HPP_

#include "columns/BaseObject.hpp"
#include "observerpattern/Subject.hpp"

namespace PV {

class HyPerCol;

/**
 * The base class for layers, connections, probes, and components of those
 * objects. Provides common interfaces for CommunicateInitInfo, AllocateDataStructures,
 * SetInitialValues messages, and a few others.
 */
class ComponentBasedObject : public BaseObject, public Subject {
  public:
   virtual ~ComponentBasedObject();

  protected:
   ComponentBasedObject();
   void initialize(char const *name, PVParams *params, Communicator *comm);

  private:
   int initialize_base();
}; // class ComponentBasedObject

} // namespace PV

#endif /* COMPONENTBASEDOBJECT_HPP_ */
