/*
 * BasePVObject.hpp
 *
 *  This is the base class for HyPerCol, layers, connections, probes, and
 *  anything else that the PV_Factory object needs to know about.
 *
 *  All objects in the BasePVObject hierarchy should have an associated
 *  instantiating function, with the prototype
 *  BasePVObject * createNameOfClass(char const * name, HyPerCol * initData);
 *
 *  Each class's instantiating function should create an object of that class,
 *  with the arguments specifying the object's name and any necessary
 *  initializing data (for most classes, this is the parent HyPerCol.
 *  For HyPerCol, it is the PVInit object).  This way, the class can be
 *  registered with the PV_Factory object by calling
 *  PV_Factory::registerKeyword() with a pointer to the class's instantiating
 *  method.
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#ifndef BASEPVOBJECT_HPP_
#define BASEPVOBJECT_HPP_

namespace PV {

class HyPerCol;

class BasePVObject {
public:
   inline char const * getName() const { return name; }
   inline HyPerCol * getParent() const { return parent; }
   char const * getKeyword() const;
   virtual ~BasePVObject();

protected:
   BasePVObject();
   int initialize(char const * name, HyPerCol * hc);
   int setName(char const * name);
   int setParent(HyPerCol * hc);

// Member variable
protected:
   char * name;
   HyPerCol * parent;

private:
   int initialize_base();
}; // class BasePVObject

BasePVObject * createBasePVObject(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* BASEPVOBJECT_HPP_ */
