/*
 * InputVolumeActivityBuffer.hpp
 */

#ifndef INPUTVOLUMEACTIVITYBUFFER_HPP_
#define INPUTVOLUMEACTIVITYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"

namespace PV {

class InputVolumeActivityBuffer : public ActivityBuffer {
  protected:
   /**
    * List of parameters used by the InputActivityBuffer class
    * @name InputLayer Parameters
    * @{
    */

   /**
    * displayPeriod: the number of timesteps each input volume is displayed before switching to
    * the next volume. If this is <= 0 or inputPath does not end in .txt, assumes the input is a
    * single file and will not change. Default value is 0.
    */
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   InputVolumeActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~InputVolumeActivityBuffer() {}

   /** Return the number of timesteps an input file is displayed before retrieveing the next
    *  volume. If the display period is zero or negative, the input never changes.
    */
   int getDisplayPeriod() const { return mDisplayPeriod; }

  protected:
   InputVolumeActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

  protected:
   // Number of timesteps an input file is displayed before advancing the file list. If <= 0, the
   // input never changes.
   int mDisplayPeriod = 0;
}; // class InputVolumeActivityBuffer

} // namespace PV

#endif // INPUTVOLUMEACTIVITYBUFFER_HPP_
