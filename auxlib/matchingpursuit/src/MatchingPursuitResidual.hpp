/*
 * MatchingPursuitResidual.hpp
 *
 *  Created on: Aug 13, 2013
 *      Author: pschultz
 */

#ifndef MATCHINGPURSUITRESIDUAL_HPP_
#define MATCHINGPURSUITRESIDUAL_HPP_

#include <layers/ANNLayer.hpp>
#include <layers/Movie.hpp>

namespace PVMatchingPursuit {

class MatchingPursuitResidual: public PV::ANNLayer {
public:
   MatchingPursuitResidual(const char * name, PV::HyPerCol * hc);
   virtual bool needUpdate(double time, double dt);
   virtual int updateState(double timed, double dt);
   virtual ~MatchingPursuitResidual();

protected:
   MatchingPursuitResidual();
   int initialize(const char * name, PV::HyPerCol * hc);

private:
   int initialize_base();

// Member variables
protected:
   bool inputInV; // set to false on initialization when trigger layer triggers; set to true after updateState loads GSynExc into V.
}; // end class MatchingPursuitResidual


PV::BaseObject * createMatchingPursuitResidual(char const * name, PV::HyPerCol * hc);

} /* namespace PVMatchingPursuit */
#endif /* MATCHINGPURSUITRESIDUAL_HPP_ */
