/*
 * Subject.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: pschultz
 */

#ifndef SUBJECT_HPP_
#define SUBJECT_HPP_

#include "observerpattern/BaseMessage.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

/**
 * The subject class of the observer pattern.
 * A class that inherits from the Subject class can maintain a hierarchy of BaseObject-derived
 * objects, and can send each object a BaseMessage-derived message, or vector of such messages,
 * using the notify() method. See the documentation of the notify() method for the expected
 * behavior of the objects in the hierarchy, and the meaning of the notify() method's return values.
 *
 * The notifyLoop methods() call notify() in a loop until notify() returns a value other than
 * PV_SUCCESS. If the return value is PV_COMPLETED, notify() returns to the caller.
 * Otherwise it exits with a fatal error.
 *
 */
class Subject {
  public:
   // No public constructors; only subclasses may be instantiated.
   virtual ~Subject();

   /**
    * Adds an Observer-derived object to the observer table.
    * Exits with an error if the object is unable to be added.
    */
   void addObserver(std::string const &tag, Observer *observer);

   ObserverTable const *getTable() { return mTable; }

  protected:
   /**
    * The default constructor called by derived classes. Derived classes should
    * call Subject::initializeTable().
    */
   Subject();

   void initializeTable(char const *tableDescription);

   /**
    * The virtual method for populating the ObserverTable data member, called by initialize().
    */
   virtual void fillComponentTable() {}

   /**
    * This method calls the respond() method of each object in the given table, using the given
    * vector of messages. If the table consists of objects A, B, and C; and the messages vector
    * consists of messages X and Y, the order is
    * A->X,
    * A->Y,
    * B->X,
    * B->Y,
    * C->X,
    * C->Y.
    *
    * The objects' respond() method returns one of the Response::Status types:
    * SUCCESS, NO_ACTION, PARTIAL, or PV_POSTPONE.
    *
    * SUCCESS: the object completed the task requested by the messages.
    * NO_ACTION: the object has nothing to do in response to the message, or had already done it.
    * PARTIAL: the object has not yet completed but is making progress. It is expected that
    * there are a small number of descrete tasks, so that an object will not return PARTIAL
    * a large number of times in response to the same message.
    * POSTPONE: the object needs to act but cannot do so until an event outside its control occurs.
    *
    * If all objects return NO_ACTION, then notify() returns NO_ACTION.
    * If all objects return either SUCCESS or NO_ACTION and there is at least one SUCCESS,
    * then notify() returns SUCCESS.
    * If all objects return either POSTPONE or NO_ACTION and there is at least one POSTPONE,
    * then notify() returns POSTPONE.
    * Otherwise, notify() returns PARTIAL.
    *
    * Generally each message in the messages vector is sent to each object in the table.
    * However, if an object returns POSTPONE in response to a message, the loop skips to
    * the next object, and does not sent any remaining messages to the postponing object.
    *
    * The rationale behind these rules is so that if the objects in the table are themselves
    * derived from the Subject class, the messages can be passed down the tree and the
    * return values passed up it, and the return value at the top can be interpreted as being
    * over all the components at the bottom, without the topmost object needing to know
    * details of the composition of the objects below it.
    *
    * If printFlag is true, the method prints information regarding postponement to standard output.
    */
   Response::Status
   notify(std::vector<std::shared_ptr<BaseMessage const>> messages, bool printFlag);

   /**
    *
    * A convenience overload of the basic notify method where there is only one message to send to
    * the objects. This overloading handles enclosing the message in a vector of length one.
    */
   inline Response::Status notify(std::shared_ptr<BaseMessage const> message, bool printFlag) {
      return notify(std::vector<std::shared_ptr<BaseMessage const>>{message}, printFlag);
   }

   /**
    * This method calls the notify() method in a loop until the result is not PARTIAL.
    * If it is either POSTPONE, it exits with a fatal error. The description argument is used in
    * the error message to report which message vector failed.
    * If the result is PV_SUCCESS, notifyLoop() returns to the calling function.
    *
    * notifyLoop should only be used if the objects that might cause a postponement are themselves
    * in the table of objects; otherwise the routine will hang.
    */
   void notifyLoop(
         std::vector<std::shared_ptr<BaseMessage const>> messages,
         bool printFlag,
         std::string const &description);

   /**
    * A convenience overload of the basic notifyLoop method where there is only one message to send
    * to the objects. This overloading handles enclosing the message in a vector of length one.
    */
   inline void notifyLoop(
         std::shared_ptr<BaseMessage const> message,
         bool printFlag,
         std::string const &description) {
      notifyLoop(std::vector<std::shared_ptr<BaseMessage const>>{message}, printFlag, description);
   }

   void deleteTable();

  protected:
   ObserverTable *mTable = nullptr;
};

} /* namespace PV */

#endif /* SUBJECT_HPP_ */
