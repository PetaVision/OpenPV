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

class Subject {
  public:
   Subject() {}
   virtual ~Subject() {}
   virtual void addObserver(Observer *observer, BaseMessage const &message) { return; }

  protected:
   void notify(
         ObserverTable const &table,
         std::vector<std::shared_ptr<BaseMessage const>> messages,
         bool printFlag);
   inline void
   notify(ObserverTable const &table, std::shared_ptr<BaseMessage const> message, bool printFlag) {
      notify(table, std::vector<std::shared_ptr<BaseMessage const>>{message}, printFlag);
   }
};

} /* namespace PV */

#endif /* SUBJECT_HPP_ */
