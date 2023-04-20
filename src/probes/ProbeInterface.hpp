#ifndef PROBEINTERFACE_HPP_
#define PROBEINTERFACE_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/BaseObject.hpp"
#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeData.hpp"
#include <memory>
#include <vector>

namespace PV {

class ProbeInterface : public BaseObject {
  public:
   typedef ProbeData<double> LayerProbeData;
   typedef std::vector<double>::size_type batchwidth_type;

   ProbeInterface(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ProbeInterface() {}

   double getCoefficient() const { return mCoefficient; }
   int getNumValues() const { return mNumValues; }
   std::vector<double> const &getValues() const;
   std::vector<double> const &getValues(double timestamp);

  protected:
   ProbeInterface() {}

   virtual void calcValues(double timestamp) = 0;

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual void initMessageActionMap() override;

   Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   Response::Status respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message);

   void setValues(ProbeData<double> const &newValues);
   void setValues(double timestamp, std::vector<double> const &newValues);

   /**
    * Sets the number of values in the ProbeData structure. Should be called during the
    * AllocateDataStructures stage.
    */
   void setNumValues(int numValues);
   void setCoefficient(double coefficient) { mCoefficient = coefficient; }

  private:
   double mCoefficient = 1.0;
   int mNumValues      = -1;

   std::shared_ptr<ProbeData<double>> mValues = nullptr;
};

} // namespace PV

#endif // PROBEINTERFACE_HPP_
