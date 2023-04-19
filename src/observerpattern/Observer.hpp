/*
 * Observer.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: pschultz
 */

#ifndef OBSERVER_HPP_
#define OBSERVER_HPP_

#include "observerpattern/BaseMessage.hpp"
#include "observerpattern/Response.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace PV {

class Observer {
  public:
   // No public constructor; only derived classes of observer can be constructed.
   virtual ~Observer() {}
   Response::Status respond(std::shared_ptr<BaseMessage const> message);
   inline std::string const &getDescription() const { return mDescription; }
   inline char const *getDescription_c() const { return mDescription.c_str(); }

  protected:
   Observer() {}
   void setDescription(std::string const &description) { mDescription = description; }
   void setDescription(char const *description) { mDescription = description; }

   /**
    * Calls the initMessageActionMap() method. Note that this function is not called
    * by the Observer's constructor; instead it must be called by the derived class's
    * initialize function. The reason is so that when initMessageActionMap is called,
    * the object uses the derived class's initMessageActionMap, not the base class's
    * method.
    */
   int initialize();

   virtual void initMessageActionMap() {}

   // Data members
  protected:
   std::map<std::string, std::function<Response::Status(std::shared_ptr<BaseMessage const>)>>
         mMessageActionMap;

  private:
   std::string mDescription;
};

} /* namespace PV */

#endif /* OBSERVER_HPP_ */
