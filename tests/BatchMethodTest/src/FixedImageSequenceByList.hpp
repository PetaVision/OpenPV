#ifndef FIXEDIMAGESEQUENCEBYLIST_HPP_
#define FIXEDIMAGESEQUENCEBYLIST_HPP_

#include "FixedImageSequence.hpp"

class FixedImageSequenceByList : public FixedImageSequence {
  public:
   FixedImageSequenceByList(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm);
   virtual ~FixedImageSequenceByList() {}

  protected:
   FixedImageSequenceByList() {}
   void initialize(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm);
   virtual void defineImageSequence() override;
}; // end class FixedImageSequenceByList

#endif // FIXEDIMAGESEQUENCEBYLIST_HPP_
