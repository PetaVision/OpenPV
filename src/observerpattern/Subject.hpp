/*
 * Subject.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: pschultz
 */

#ifndef SUBJECT_HPP_
#define SUBJECT_HPP_

#include "observerpattern/ObserverTable.hpp"
#include "observerpattern/BaseMessage.hpp"

namespace PV {

class Subject {
public:
   Subject() {}
   virtual ~Subject() {}
   virtual void addObserver(Observer * observer, BaseMessage const& message) { return; }
protected:
   void notify(ObserverTable const& table, std::vector<std::shared_ptr<BaseMessage const> > messages);
   inline void notify(ObserverTable const& table, std::shared_ptr<BaseMessage const> message) {
      notify(table, std::vector<std::shared_ptr<BaseMessage const> >{message});
   }

};

} /* namespace PV */

#endif /* SUBJECT_HPP_ */
