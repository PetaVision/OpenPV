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
   TextStreamProbe(const char * probeName, HyPerCol * hc);
   virtual ~TextStreamProbe();

   virtual int communicateInitInfo();

   virtual int outputState(double timef);

protected:
   TextStreamProbe();
   int initTextStreamProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);
   void featureNumberToCharacter(int code, char ** cbufptr, char * bufstart, int buflen);

private:
   int initTextStreamProbe_base();

   bool useCapitalization;
   double displayPeriod;
   double nextDisplayTime;
};

} /* namespace PV */
#endif /* TEXTSTREAMPROBE_HPP_ */
