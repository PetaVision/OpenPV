#ifndef FIXEDIMAGESEQUENCEBYFILE_HPP_
#define FIXEDIMAGESEQUENCEBYFILE_HPP_

#include "FixedImageSequence.hpp"

class FixedImageSequenceByFile : public FixedImageSequence {
  public:
   FixedImageSequenceByFile(char const *name, PV::HyPerCol *hc);
   virtual ~FixedImageSequenceByFile() {}

  protected:
   FixedImageSequenceByFile() {}
   int initialize(char const *name, PV::HyPerCol *hc);
   virtual void defineImageSequence();
}; // end class FixedImageSequenceByFile

#endif // FIXEDIMAGESEQUENCEBYFILE_HPP_
