#ifndef BUFFERPARAMINTERFACE_HPP_
#define BUFFERPARAMINTERFACE_HPP_

#include "io/PVParams.hpp"
#include "probes/ProbeComponent.hpp"
#include "probes/StatsProbeTypes.hpp"
#include <string>

namespace PV {

/**
 * BufferParamInterface is a pure virtual method for the interface for reading
 * a parameter string buffer into a StatsBufferType (V or A).
 * Implementing classes must override ioParam_buffer().
 */
class BufferParamInterface : public ProbeComponent {
  public:
   virtual ~BufferParamInterface();

   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) = 0;

   StatsBufferType getBufferType() const { return mBufferType; }

  protected:
   BufferParamInterface() {}

   void initialize(char const *name, PVParams *params);

   /**
    * A method for reading BufferString from params or writing BufferString to
    * a params output file, based on the value of ioFlag.
    * Implementing classes will still need to call setBufferType() in the case
    * where ioFlag is set to READ.
    *
    * It is provided here so that BufferString may remain a private data member,
    * with this method as the interface for interacting with the params.
    */
   void internal_ioParam_buffer(enum ParamsIOFlag ioFlag);

   StatsBufferType parseBufferType(char const *bufferString);

   char *getBufferString() { return mBufferString; }
   char const *getBufferString() const { return mBufferString; }

   /**
    * Sets the BufferType data member to the indicated type, and sets the
    * BufferString data member to either "Activity" or "Membrane Potential"
    * accordingly.
    */
   void setBufferType(StatsBufferType bufferType);

  private:
   char *mBufferString = nullptr;
   StatsBufferType mBufferType;
};

} // namespace PV

#endif // BUFFERPARAMINTERFACE_HPP_
