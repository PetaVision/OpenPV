#ifndef FIXEDIMAGESEQUENCEBYFILE_HPP_
#define FIXEDIMAGESEQUENCEBYFILE_HPP_

#include "FixedImageSequence.hpp"

class FixedImageSequenceByFile : public FixedImageSequence {
  public:
   FixedImageSequenceByFile(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm);
   virtual ~FixedImageSequenceByFile() {}

  protected:
   FixedImageSequenceByFile() {}
   void initialize(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm);
   virtual void defineImageSequence() override;
}; // end class FixedImageSequenceByFile

#endif // FIXEDIMAGESEQUENCEBYFILE_HPP_
