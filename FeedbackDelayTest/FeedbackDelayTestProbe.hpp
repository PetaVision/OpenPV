/*
 * FeedbackDelayTestProbe.hpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#ifndef FEEDBACKDELAYTESTPROBE_HPP_
#define FEEDBACKDELAYTESTPROBE_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class FeedbackDelayTestProbe: public PV::StatsProbe {
public:
   FeedbackDelayTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
   FeedbackDelayTestProbe(HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);
protected:
   bool toggleOutput;

};

} /* namespace PV */
#endif /* FEEDBACKDELAYTESTPROBE_HPP_ */
