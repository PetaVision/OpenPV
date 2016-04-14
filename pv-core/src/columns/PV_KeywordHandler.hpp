/*
 * PV_KeywordHandler.hpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#ifndef PV_KEYWORDHANDLER_HPP_
#define PV_KEYWORDHANDLER_HPP_

#include <columns/BasePVObject.hpp>

namespace PV {

typedef BasePVObject * (*ObjectCreateFn)(char const * name, HyPerCol * hc);

class HyPerCol;

/**
 * PV_KeywordHandler is a class that associates a string, the keyword,
 * with a function pointer for creating objects of a type corresponding
 * to that keyword.
 * It is used by PV_Factory to to manage creating layers, connections,
 * etc., within a HyPerCol.  It is generally not used except
 * internally by PV_Factory.
 *
 * As an example, after creating a PV_KeywordHandler with the statement
 *
 * PV_KeywordHandler * kwh = new PV_KeywordHandler("HyPerLayer", createHyPerLayer);
 *
 * (the function createHyPerLayer is defined in layers/HyPerLayer.hpp),
 * one can insert a new HyPerLayer called "layer" into a given HyPerCol with
 * the statement.
 * 
 * kwh->create("layer", hc);
 */
class PV_KeywordHandler {
public:
   /**
    * The constructor for PV_KeywordHandler.
    */
   PV_KeywordHandler(char const * kw, ObjectCreateFn creator);

   /**
    * The copy constructor for PV_KeywordHandler
    */
   PV_KeywordHandler(PV_KeywordHandler const& orig);

   /**
    * The copy assignment operator for PV_KeywordHandler
    */
   PV_KeywordHandler& operator=(PV_KeywordHandler const& orig);

   /**
    * A get-method for the handler's keyword.
    */
   const char * getKeyword() const { return keyword; }

   /**
    * A get-method for the handler's creator function pointer.
    */
   ObjectCreateFn getCreator() const { return creator; }

   /**
    * The method that calls the function pointer with the given arguments
    */
   BasePVObject * create(char const * name, HyPerCol * hc) const;

   /**
    * The destructor for PV_KeywordHandler.
    */
   virtual ~PV_KeywordHandler();

protected:
   /**
    * A method used internally by the constructors and copy assignment operator
    * to set the initialize the PV_KeywordHandler object.
    */
   int initialize(char const * kw, BasePVObject * (*creator)(char const * name, HyPerCol * hc));

// Member variables
private:
   char * keyword;
   ObjectCreateFn creator;
};

} /* namespace PV */

#endif /* PV_KEYWORDHANDLER_HPP_ */
