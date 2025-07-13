#ifndef BUFFERPARAMINTERFACE_HPP_
#define BUFFERPARAMINTERFACE_HPP_

#include "params/ParamsIO.hpp"
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

   virtual void ioParam_buffer(ParamsIOSwitch ioSwitch) = 0;

   StatsBufferType getBufferType() const { return mBufferType; }

  protected:
   BufferParamInterface() {}

   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);

   /**
    * A method for reading BufferString from params or writing BufferString to
    * a params output file, based on the value of ioSwitch.
    * Implementing classes will still need to call setBufferType() in the case
    * where ioSwitch is set to Read.
    *
    * It is provided here so that BufferString may remain a private data member,
    * with this method as the interface for interacting with the params.
    */
   void internal_ioParam_buffer(ParamsIOSwitch ioSwitch);

   StatsBufferType parseBufferType(std::string const &bufferString);

   std::string const &getBufferString() const { return mBufferString; }

   /**
    * Sets the BufferType data member to the indicated type, and sets the
    * BufferString data member to either "Activity" or "Membrane Potential"
    * accordingly.
    */
   void setBufferType(StatsBufferType bufferType);

  private:
   std::string mBufferString;
   StatsBufferType mBufferType;
};

} // namespace PV

#endif // BUFFERPARAMINTERFACE_HPP_
