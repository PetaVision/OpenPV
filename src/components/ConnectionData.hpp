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
   /** @} */ // end of ConnectionData parameters

  public:
   ConnectionData(char const *name, HyPerCol *hc);
   virtual ~ConnectionData();

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

  protected:
   ConnectionData();

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

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

  protected:
   char *mPreLayerName  = nullptr;
   char *mPostLayerName = nullptr;
   HyPerLayer *mPre     = nullptr;
   HyPerLayer *mPost    = nullptr;

}; // class ConnectionData

} // namespace PV

#endif // CONNECTIONDATA_HPP_
