/*
 * Factory.hpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#ifndef FACTORY_HPP_
#define FACTORY_HPP_

#include <columns/KeywordHandler.hpp>
#include <cstddef>
#include <vector>

namespace PV {

class BaseObject;

/**
 * The class to generate layers, connections, etc. for adding to a HyPerCol.
 * The function build() in buildandrun.cpp (which underlies all the
 * functions in buildandrun.cpp) uses Factory to build the HyPerCol.
 * Factory is a singleton which is retrieved using the static method
 * Factory::instance().
 *
 * The standard Factory constructor already registers the objects in the
 * PetaVision core.  For example, calling Factory::instance()->create()
 * method with keyword "ANNLayer" returns a new ANNLayer.
 *
 * If you have a custom class, you need to register it with the Factory
 * object by associating its keyword with a function instantiates a new
 * object of that class, and returns it as a pointer of type BaseObject
 * In most cases, the template Factory::create is sufficient
 * as the instantiator function.
 *
 * For example:
 *
 * class CustomLayerType : public HyPerLayer {
 * ...
 * };
 * ...
 * PV_Init pv_init(&argc, &argv, false);
 * HyPerCol * hc = new HyPerCol("column", &pv_init);
 * pv_init.registerKeyword("CustomType", Factory::create<CustomType>);
 * CustomType customObject = *Factory::instance()->create("CustomType", params, comm);
 * ...
 * Note that buildandrun() automates the task of calling the create() method;
 * in practice, you only need to specify the instantiator function, and
 * call the registerKeyword method calling before one of the buildandrun
 * functions.
 *
 * It is possible to use a custom instantiator function instead of create.
 * The function must take two arguments, the name as a C-style constant string
 * and
 * a pointer to a HyPerCol.
 *
 * For example:
 * ...
 * BaseObject * createCustomLayerType(char const *name, PVParams *params, Communicator *comm) {
 *    return new CustomLayerType(name, params, comm);
 * }
 * ...
 * pv_init.registerKeyword("CustomLayerType", createCustomLayerType);
 * ...
 */
class Factory {
  public:
   static Factory *instance() {
      static Factory *singleton = new Factory();
      return singleton;
   }

   /**
    * A function template that can be used to register most subclasses of
    * BaseObject in the factory using the registerKeyword function.  The
    * requirements on the BaseObject subclass is that it have a constructor
    * with two arguments, the name and a pointer to the HyPerCol.
    */
   template <typename T>
   static BaseObject *create(char const *name, PVParams *params, Communicator *comm) {
      return new T(name, params, comm);
   }

   /**
    * The method to add a new object type to the Factory.
    * keyword is the string that labels the object type, matching the keyword
    * used in params files.
    * creator is a pointer to a function that takes a name and a HyPerCol
    * pointer, and
    * creates an object of the corresponding keyword, with the given name and
    * parent HyPerCol.
    * The function should return a pointer of type BaseObject, created with the
    * new operator.
    */
   int registerKeyword(char const *keyword, ObjectCreateFn creator);

   /**
    * The method to create an object of the type specified by keyword, with the
    * given name
    * and parent HyPerCol.  It calls the function associated with the keyword by
    * the
    * registerKeyword pointer.
    */
   BaseObject *
   createByKeyword(char const *keyword, char const *name, PVParams *params, Communicator *comm)
         const;

  private:
   /**
    * The constructor for Factory.  It initializes the list of known keywords to
    * the core PetaVision
    * keywords.
    */
   Factory();

   /**
    * The destructor for Factory
    */
   virtual ~Factory();

   /**
    * The function called by the default constructor, to add the core PetaVision
    * keywords.
    */
   int registerCoreKeywords();

   /**
    * A method used internally by the copy assignment operator and copy
    * constructor,
    * to copy a keyword handler list into the Factory.
    */
   int copyKeywordHandlerList(std::vector<KeywordHandler *> const &orig);

   /**
    * A method used internally to retrieve the keyword handler corresponding to a
    * given keyword.
    */
   KeywordHandler const *getKeywordHandler(char const *keyword) const;

   /**
    * A method used internally by the copy assignment operator and destructor, to
    * deallocate and clear the keyword handler list.
    */
   int clearKeywordHandlerList();

   // Member variables
  private:
   std::vector<KeywordHandler *> keywordHandlerList;
}; // class Factory

} // namespace PV

#endif /* FACTORY_HPP_ */
