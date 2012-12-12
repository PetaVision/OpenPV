/*
 * LeakyIntegratedCountLayer.hpp
 *
 *  Created on: Dec 6, 2012
 *      Author: pschultz
 */

#ifndef LEAKYINTEGRATEDCOUNTLAYER_HPP_
#define LEAKYINTEGRATEDCOUNTLAYER_HPP_

#include "ANNLayer.hpp"
#include "../include/pv_types.h"

namespace PV {

class LeakyIntegratedCountLayer: public PV::ANNLayer {

// Methods
public:
   LeakyIntegratedCountLayer(const char * name, HyPerCol * hc);
   virtual ~LeakyIntegratedCountLayer();
   virtual int updateState(double timed, double dt);

   // get-methods
   double getDecayTime() {return decayTime;}

protected:
   LeakyIntegratedCountLayer();
   int initialize(const char * name, HyPerCol * hc);

   virtual int readVThreshParams(PVParams * params); // V thresholds not used in LeakyIntegratedCountLayer, so the params are set to infinite values

   // Methods for loading member variables from params file.  Declared virtual so that a derived classes can deactivate a param if it isn't needed.
   virtual float readIntegrationTime() {return getParent()->parameters()->value(name, "decayTime", decayTime);}

private:
   int initialize_base();

// Member variables
protected:
   double decayTime;
}; // end of class LeakyIntegratedCountLayer

} /// end namespace PV

#endif /* LEAKYINTEGRATEDCOUNTLAYER_HPP_ */
