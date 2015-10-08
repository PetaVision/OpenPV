/*
 * TextStreamProbe.hpp
 *
 *  Created on: May 20, 2013
 *      Author: pschultz
 */

#ifndef TEXTSTREAMPROBE_HPP_
#define TEXTSTREAMPROBE_HPP_

#include <io/LayerProbe.hpp>

namespace PVtext {

class TextStreamProbe: public PV::LayerProbe {
public:
   TextStreamProbe(const char * probeName, PV::HyPerCol * hc);
   virtual ~TextStreamProbe();

   virtual int communicateInitInfo();

   virtual int outputState(double timef);

protected:
   TextStreamProbe();
   int initTextStreamProbe(const char * probeName, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);
   void featureNumberToCharacter(int code, char ** cbufptr, char * bufstart, int buflen);
   virtual int initNumValues();
   virtual int calcValues(double timevalue);

private:
   int initTextStreamProbe_base();

   bool useCapitalization;
   double displayPeriod;
   double nextDisplayTime;
};

} /* namespace PVtext */
#endif /* TEXTSTREAMPROBE_HPP_ */
