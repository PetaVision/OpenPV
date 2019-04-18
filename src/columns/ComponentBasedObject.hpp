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

/**
 * The base class for layers, connections, ActivityComponent, and other classes that
 * comprise components. This class inherits the Subject class to provide a component
 * table.
 */
class ComponentBasedObject : public BaseObject, public Subject {
  public:
   virtual ~ComponentBasedObject();

  protected:
   /** The default constructor for ComponentBasedObject does nothing. Derived classes
    *  should call ComponentBasedObject::initialize() during their own initialization.
    */
   ComponentBasedObject();
   void initialize(char const *name, PVParams *params, Communicator const *comm);

   /**
     * When called with the write flag, calls the ioParams function of each component.
     * When called with the read flag, does nothing since components read their params
     * during instantiation.
     */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
}; // class ComponentBasedObject

} // namespace PV

#endif /* COMPONENTBASEDOBJECT_HPP_ */
