/*
 * LinkedObjectParam.hpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#ifndef LINKEDOBJECTPARAM_HPP_
#define LINKEDOBJECTPARAM_HPP_

#include "columns/BaseObject.hpp"
#include "columns/ComponentBasedObject.hpp"

namespace PV {

/**
 * The base class for components that read a single string parameter that points to
 * another object in the hierarchy.
 * Examples:
 *     OriginalLayerNameParam, used by CloneVLayer, RescaleLayer, etc.
 *     OriginalConnNameParam, used by CloneConn, TransposeConn, etc.
 * The object name (the param's value) is retrieved using the getLinkedObjectName() method.
 * The object itself is retrieved from an ObserverTable (as a pointer to a
 * ComponentBasedObject) using the findLinkedObject function member.
 */
class LinkedObjectParam : public BaseObject {
  protected:
   /**
    * List of parameters needed from the LinkedObjectParam class
    * @name LinkedObjectParam Parameters
    * @{
    */

   /**
    * @brief The LinkedObjectParam constructor contains an additional string argument
    * that specifies the param name to search for. This method reads and writes
    * the param whose name is specified by that argument.
    */
   virtual void ioParam_linkedObjectName(enum ParamsIOFlag ioFlag);

   /** @} */ // end of LinkedObjectParam parameters

  public:
   virtual ~LinkedObjectParam();

   char const *getLinkedObjectName() const { return mLinkedObjectName; }

   /**
    * Searches the given object map for an object whose name matches
    * the LinkedObjectName. If there is no such object, or if the object
    * is not a ComponentBasedObject or a ComponentBasedObject-derived class,
    * an invalid argument exception is thrown, with the message
    * <ParamName> "<LinkedObjectName>" does not correspond to an object in the column.
    */
   ComponentBasedObject *findLinkedObject(ObserverTable const *hierarchy);

  protected:
   LinkedObjectParam() {}

   void initialize(
         char const *name,
         PVParams *params,
         Communicator const *comm,
         std::string const &paramName);

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   std::string mParamName;
   char *mLinkedObjectName = nullptr;
};

} // namespace PV

#endif // LINKEDOBJECTPARAM_HPP_
