/*
 * PV_Factory.hpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#ifndef PV_FACTORY_HPP_
#define PV_FACTORY_HPP_

#include <columns/PV_KeywordHandler.hpp>
#include <vector>
#include <cstddef>

namespace PV {

class BasePVObject;

/**
 * The class to manage adding layers, connections, etc. to a HyPerCol.
 * The function build() in buildandrun.cpp (which is the basis of all the
 * functions in buildandrun.cpp) uses PV_Factory to build the HyPerCol.
 *
 * This class is accessible only by the PV_Init class, which has the
 * member variable pv_factory.
 *
 * The standard PV_Factory constructor already registers the objects
 * in the PetaVision core.  For example, calling the PV_Factory::create()
 * method with keyword "ANNLayer" calls the createANNLayer() object
 * creator function in layers/ANNLayer.cpp.  In practice, the
 * PV_Factory::create() method is called by the PV_Init::create() method,
 * which passes the arguments to the PV_Factory:
 *
 * PV_Init pv_init(&argc, &argv, false); // 3rd argument is whether to allow command-line arguments that PV doesn't recognize
 * HyPerCol * hc = new HyPerCol("column", pv_init);
 * pv_init.create("ANNLayer", "layer", hc);
 *
 * (note that the functions in buildandrun.cpp and the PV_Init::build() method
 * automate the task of calling the create() method for the groups in the
 * params file).
 *
 * If you have a custom object, you need to register it with the PV_Factory
 * object by associating its keyword with a function pointer that takes
 * two arguments: a C-style string giving the name of the individual layer,
 * and pointer to a HyPerCol.  This function should instantiate the new object
 * with the new operator, and return it as a pointer of type BasePVObject
 * the base class for layers, connections, etc.)  In practice, the
 * PV_Factory::registerKeyword() method is called by the PV_registerKeyword()
 * method, which again passes its arguments to the PV_Factory.
 * For example:
 *
 * class CustomLayerType : public HyPerLayer {
 * ...
 * };
 * ...
 * BasePVObject * createCustomLayerType(char const * name, HyPerCol * hc) {
 *    return new CustomLayerType(name, hc);
 * }
 * ...
 * PV_Init pv_init(&argc, &argv, false);
 * HyPerCol * hc = new HyPerCol("column", &pv_init);
 * pv_init.registerKeyword("customLayerType", createCustomLayerType);
 * pv_init.create("customLayerType", hc)
 * ...
 *
 * Again, the functions in buildandrun.cpp automate the task of calling the create() method;
 * in practice, you only need define the creator function createCustomLayerType, and
 * call the registerKeyword method calling before one of the buildandrun functions.
 */
class PV_Factory {
   friend class PV_Init;

public:
   /**
    * A function pointer that can be used in a registerKeyword call, that always returns the
    * null object.  The "normalizeGroup", "none", and "" keywords all use the createNull function pointer.
    */
   static BasePVObject * createNull(char const * name, HyPerCol * hc) {return NULL;}

private:
   /**
    * The constructor for PV_Factory.  It initializes the list of known keywords to the core PetaVision keywords.
    */
   PV_Factory();

   /**
    * The copy constructor for PV_Factory
    */
   PV_Factory(PV_Factory const& orig);

   /**
    * The copy assignment operator for PV_Factory
    */
   PV_Factory& operator=(PV_Factory const& orig);

   /**
    * The method to add a new object type to the PV_Factory.
    * keyword is the string that labels the object type, matching the keyword used in params files.
    * creator is a pointer to a function that takes a name and a HyPerCol pointer, and
    * creates an object of the corresponding keyword, with the given name and parent HyPerCol.
    * The function should return a pointer of type BasePVObject, created with the new operator.
    */
   int registerKeyword(char const * keyword, ObjectCreateFn creator);

   /**
    * The method to create an object of the type specified by keyword, with the given name
    * and parent HyPerCol.  It calls the function associated with the keyword by the
    * registerKeyword pointer.
    */
   BasePVObject * create(char const * keyword, char const * name, HyPerCol * hc) const;

   /**
    * The destructor for PV_Factory
    */
   virtual ~PV_Factory();

   /**
    * The function called by the default constructor, to add the core PetaVision keywords.
    */
   int registerCoreKeywords();

   /**
    * A method used internally by the copy assignment operator and copy constructor,
    * to copy a keyword handler list into the PV_Factory.
    */
   int copyKeywordHandlerList(std::vector<PV_KeywordHandler*> const& orig);

   /**
    * A method used internally to retrieve the keyword handler corresponding to a given keyword.
    */
   PV_KeywordHandler const * getKeywordHandler(char const * keyword) const;

   /**
    * A method used internally by the copy assignment operator and destructor, to
    * deallocate and clear the keyword handler list.
    */
   int clearKeywordHandlerList();

// Member variables
private:
   std::vector<PV_KeywordHandler *>keywordHandlerList;
}; // class PV_Factory

}  // namespace PV

#endif /* PV_FACTORY_HPP_ */
