#ifndef IMAGEBYLISTUSINGTIMESTAMP_HPP_
#define IMAGEBYLISTUSINGTIMESTAMP_HPP_

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

#endif // IMAGEBYLISTUSINGTIMESTAMP_HPP_
