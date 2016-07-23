/*
 * BaseObject.hpp
 *
 *  This is the base class for HyPerCol, layers, connections, probes, and
 *  anything else that the Factory object needs to know about.
 *
 *  All objects in the BaseObject hierarchy should have an associated
 *  instantiating function, with the prototype
 *  BaseObject * createNameOfClass(char const * name, HyPerCol * initData);
 *
 *  Each class's instantiating function should create an object of that class,
 *  with the arguments specifying the object's name and any necessary
 *  initializing data (for most classes, this is the parent HyPerCol.
 *  For HyPerCol, it is the PVInit object).  This way, the class can be
 *  registered with the Factory object by calling
 *  Factory::registerKeyword() with a pointer to the class's instantiating
 *  method.
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#ifndef BASEOBJECT_HPP_
#define BASEOBJECT_HPP_

#include "columns/Messages.hpp"
#include "include/pv_common.h"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVAlloc.hpp"

namespace PV {

class HyPerCol;

class BaseObject {
public:
   inline char const * getName() const { return name; }
   inline HyPerCol * getParent() const { return parent; }
   inline char const * getDescription_c() const { return description.c_str(); }
   inline std::string const& getDescription() const { return description; }
   char const * getKeyword() const;
   int respond(BaseMessage const * message); // TODO: should return enum with values corresponding to PV_SUCCESS, PV_FAILURE, PV_POSTPONE
   virtual ~BaseObject();


protected:
   BaseObject();
   int initialize(char const * name, HyPerCol * hc);
   int setName(char const * name);
   int setParent(HyPerCol * hc);
   virtual int setDescription();

   virtual int respondConnectionUpdate(ConnectionUpdateMessage const * message) { return PV_SUCCESS; }
   virtual int respondConnectionFinalizeUpdate(ConnectionFinalizeUpdateMessage const * message) { return PV_SUCCESS; }
   virtual int respondConnectionOutput(ConnectionOutputMessage const * message) { return PV_SUCCESS; }
   virtual int respondLayerPublish(LayerPublishMessage const * message) { return PV_SUCCESS; }

// Member variable
protected:
   char * name = nullptr;
   HyPerCol * parent = nullptr;
   std::string description;

private:
   int initialize_base();
}; // class BaseObject

BaseObject * createBasePVObject(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* BASEOBJECT_HPP_ */
