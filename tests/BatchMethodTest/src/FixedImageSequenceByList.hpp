#ifndef IMAGEBYLISTUSINGTIMESTAMP_HPP_
#define IMAGEBYLISTUSINGTIMESTAMP_HPP_

#include "FixedImageSequence.hpp"

class FixedImageSequenceByList : public FixedImageSequence {
  public:
   FixedImageSequenceByList(char const *name, PV::HyPerCol *hc);
   virtual ~FixedImageSequenceByList() {}

  protected:
   FixedImageSequenceByList() {}
   int initialize(char const *name, PV::HyPerCol *hc);
   virtual void defineImageSequence();
}; // end class FixedImageSequenceByList

#endif // IMAGEBYLISTUSINGTIMESTAMP_HPP_
