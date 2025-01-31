#ifndef FIXEDIMAGESEQUENCEBYLIST_HPP_
#define FIXEDIMAGESEQUENCEBYLIST_HPP_

#include "FixedImageSequence.hpp"

class FixedImageSequenceByList : public FixedImageSequence {
  public:
   FixedImageSequenceByList(char const *name, PV::PVParams *params, PV::Communicator const *comm);
   virtual ~FixedImageSequenceByList() {}

  protected:
   FixedImageSequenceByList() {}
   void initialize(char const *name, PV::PVParams *params, PV::Communicator const *comm);
   virtual void defineImageSequence() override;
}; // end class FixedImageSequenceByList

#endif // FIXEDIMAGESEQUENCEBYLIST_HPP_
