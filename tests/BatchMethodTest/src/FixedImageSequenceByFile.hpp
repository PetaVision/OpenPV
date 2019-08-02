#ifndef FIXEDIMAGESEQUENCEBYFILE_HPP_
#define FIXEDIMAGESEQUENCEBYFILE_HPP_

#include "FixedImageSequence.hpp"

class FixedImageSequenceByFile : public FixedImageSequence {
  public:
   FixedImageSequenceByFile(char const *name, PV::PVParams *params, PV::Communicator const *comm);
   virtual ~FixedImageSequenceByFile() {}

  protected:
   FixedImageSequenceByFile() {}
   void initialize(char const *name, PV::PVParams *params, PV::Communicator const *comm);
   virtual void defineImageSequence() override;
}; // end class FixedImageSequenceByFile

#endif // FIXEDIMAGESEQUENCEBYFILE_HPP_
