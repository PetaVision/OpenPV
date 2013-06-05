/*
 * TextStreamProbe.hpp
 *
 *  Created on: May 20, 2013
 *      Author: pschultz
 */

#ifndef TEXTSTREAMPROBE_HPP_
#define TEXTSTREAMPROBE_HPP_

#include "LayerProbe.hpp"

namespace PV {

class TextStreamProbe: public PV::LayerProbe {
public:
   TextStreamProbe(const char * filename, HyPerLayer * layer, pvdata_t display_period);
   virtual ~TextStreamProbe();

   virtual int outputState(double timef);

protected:
   TextStreamProbe();
   int initTextStreamProbe(const char * filename, HyPerLayer * layer, pvdata_t display_period);
   void featureNumberToCharacter(int code, char ** cbufptr, char * bufstart, int buflen);

private:
   int initTextStreamProbe_base();

   bool useCapitalization;
   pvdata_t displayPeriod;
   double nextDisplayTime;
};

} /* namespace PV */
#endif /* TEXTSTREAMPROBE_HPP_ */
