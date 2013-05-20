/*
 * TextStreamProbe.hpp
 *
 *  Created on: May 20, 2013
 *      Author: pschultz
 */

#ifndef TEXTSTREAMPROBE_HPP_
#define TEXTSTREAMPROBE_HPP_

#include "../PetaVision/src/io/LayerProbe.hpp"

namespace PV {

class TextStreamProbe: public PV::LayerProbe {
public:
   TextStreamProbe(const char * filename, HyPerLayer * layer);
   virtual ~TextStreamProbe();

   virtual int outputState(double timef);

protected:
   TextStreamProbe();
   int initTextStreamProbe(const char * filename, HyPerLayer * layer);
   void featureNumberToCharacter(int code, char ** cbufptr, char * bufstart, int buflen);

private:
   int initTextStreamProbe_base();

   bool useCapitalization;
};

} /* namespace PV */
#endif /* TEXTSTREAMPROBE_HPP_ */
