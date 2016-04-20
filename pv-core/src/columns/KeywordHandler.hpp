/*
 * KeywordHandler.hpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#ifndef KEYWORDHANDLER_HPP_
#define KEYWORDHANDLER_HPP_

#include <columns/BaseObject.hpp>

namespace PV {

typedef BaseObject * (*ObjectCreateFn)(char const * name, HyPerCol * hc);

class HyPerCol;

/**
 * KeywordHandler is a class that associates a string, the keyword,
 * with a function pointer for creating objects of a type corresponding
 * to that keyword.
 * It is used by Factory to to manage creating layers, connections,
 * etc., within a HyPerCol.  It is generally not used except
 * internally by Factory.
 *
 * As an example, after creating a KeywordHandler with the statement
 *
 * KeywordHandler * kwh = new KeywordHandler("HyPerLayer", createHyPerLayer);
 *
 * (the function createHyPerLayer is defined in layers/HyPerLayer.hpp),
 * one can insert a new HyPerLayer called "layer" into a given HyPerCol with
 * the statement.
 * 
 * kwh->create("layer", hc);
 */
class KeywordHandler {
public:
   /**
    * The constructor for KeywordHandler.
    */
   KeywordHandler(char const * kw, ObjectCreateFn creator);

   /**
    * The copy constructor for KeywordHandler
    */
   KeywordHandler(KeywordHandler const& orig);

   /**
    * The copy assignment operator for KeywordHandler
    */
   KeywordHandler& operator=(KeywordHandler const& orig);

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
   BaseObject * create(char const * name, HyPerCol * hc) const;

   /**
    * The destructor for KeywordHandler.
    */
   virtual ~KeywordHandler();

protected:
   /**
    * A method used internally by the constructors and copy assignment operator
    * to set the initialize the KeywordHandler object.
    */
   int initialize(char const * kw, BaseObject * (*creator)(char const * name, HyPerCol * hc));

// Member variables
private:
   char * keyword;
   ObjectCreateFn creator;
};

} /* namespace PV */

#endif /* KEYWORDHANDLER_HPP_ */
