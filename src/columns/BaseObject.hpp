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

namespace PV {

class HyPerCol;

class BaseObject {
public:
   inline char const * getName() const { return name; }
   inline HyPerCol * getParent() const { return parent; }
   char const * getKeyword() const;
   virtual ~BaseObject();

protected:
   BaseObject();
   int initialize(char const * name, HyPerCol * hc);
   int setName(char const * name);
   int setParent(HyPerCol * hc);

// Member variable
protected:
   char * name;
   HyPerCol * parent;

private:
   int initialize_base();
}; // class BaseObject

BaseObject * createBasePVObject(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* BASEOBJECT_HPP_ */
