/*
 * ConnectionData.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: pschultz
 */

#ifndef CONNECTIONDATA_HPP_
#define CONNECTIONDATA_HPP_

#include "columns/BaseObject.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

class ConnectionData : public BaseObject {
  protected:
   /**
    * List of parameters needed from the ConnectionData class
    * @name ConnectionData Parameters
    * @{
    */

   /**
    * @brief preLayerName: Specifies the connection's pre layer
    * @details Required parameter
    */
   virtual void ioParam_preLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief preLayerName: Specifies the connection's post layer
    * @details Required parameter
    */
   virtual void ioParam_postLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief numAxonalArbors: Specifies the number of arbors to use in the connection
    */
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);

   /**
    * @brief delay: Specifies delay(s) which the post layer will receive data
    * @details: Delays are specified in units of dt, but are rounded to be integer multiples of dt.
    * If delay is a scalar, all arbors of the connection have that value of delay.
    * If delay is an array, the length must match the number of arbors and the arbors are assigned
    * the delays sequentially.
    */
   virtual void ioParam_delay(enum ParamsIOFlag ioFlag);

   /**
    * @brief initializeFromCheckpointFlag: If set to true, initialize using checkpoint direcgtory
    * set in HyPerCol.
    * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);
   /** @} */ // end of ConnectionData parameters

  public:
   ConnectionData(char const *name, HyPerCol *hc);
   virtual ~ConnectionData();

   virtual int setDescription() override;

   /**
    * Returns the name of the connection's presynaptic layer.
    */
   char const *getPreLayerName() const { return mPreLayerName; }

   /**
    * Returns the name of the connection's postsynaptic layer.
    */
   char const *getPostLayerName() const { return mPostLayerName; }

   /**
    * Returns a pointer to the connection's presynaptic layer.
    */
   HyPerLayer *getPre() { return mPre; }

   /**
    * Returns a pointer to the connection's postsynaptic layer.
    */
   HyPerLayer *getPost() { return mPost; }

   /**
    * Returns the number of arbors in the connection
    */
   int getNumAxonalArbors() const { return mNumAxonalArbors; }

   int getDelay(int arbor) const { return mDelay[arbor]; }

   bool getInitializeFromCheckpointFlag() const { return mInitializeFromCheckpointFlag; }

  protected:
   ConnectionData();

   int initialize(char const *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   void initializeDelays();

   void setDelay(int arborId, double delay);

   /**
    * If the character string given by name has the form "AbcToXyz", then
    * preLayerNameString is set to Abc and postLayerNameString is set to Xyz.
    * If the given character string does not contain "To" or if it contains
    * "To" in more than one place, an error message is printed and the
    * preLayerNameString and postLayerNameString are set to the empty string.
    */
   static void inferPreAndPostFromConnName(
         const char *name,
         int rank,
         std::string &preLayerNameString,
         std::string &postLayerNameString);

   int maxDelaySteps();

  protected:
   char *mPreLayerName  = nullptr;
   char *mPostLayerName = nullptr;
   HyPerLayer *mPre     = nullptr;
   HyPerLayer *mPost    = nullptr;
   int mNumAxonalArbors = 1;
   std::vector<int> mDelay; // The delays expressed in # of timesteps (delays ~= fDelayArray / t)
   double *mDelaysParams = nullptr; // The raw delays in params, in the same units that dt is in.
   int mNumDelays        = 0; // The size of the mDelayParams array

   // If this flag is set and HyPerCol sets initializeFromCheckpointDir, load initial state from
   // the initializeFromCheckpointDir directory.
   bool mInitializeFromCheckpointFlag = true;

}; // class ConnectionData

} // namespace PV

#endif // CONNECTIONDATA_HPP_
