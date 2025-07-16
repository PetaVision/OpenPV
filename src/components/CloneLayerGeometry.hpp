/*
 * CloneLayerGeometry.hpp
 */

#ifndef CLONELAYERGEOMETRY_HPP_
#define CLONELAYERGEOMETRY_HPP_

#include "components/LayerGeometry.hpp"

namespace PV {

/**
 * A component, used by CloneVLayer, that does not read parameters,
 * but uses the layer specified in the originalLayerName parameter
 * to create the PVLayerLoc.
 */
class CloneLayerGeometry : public LayerGeometry {
  protected:
   /**
    * List of parameters needed from the CloneLayerGeometry class
    * @name CloneLayerGeometry Parameters
    * @{
    */

   /**
    * @brief broadcastFlag: CloneLayerGeometry does not read the broadcastFlag parameter.
    * Instead, it uses the broadcastFlag of the original layer.
    */
   virtual void ioParam_broadcastFlag(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nxScale: CloneLayerGeometry does not read the nxScale parameter.
    * Instead, it uses the nxScale of the original layer.
    */
   virtual void ioParam_nxScale(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nyScale: CloneLayerGeometry does not read the nyScale parameter.
    * Instead, it uses the nyScale of the original layer.
    */
   virtual void ioParam_nyScale(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nf: CloneLayerGeometry does not read the nf parameter.
    * Instead, it uses the nf of the original layer.
    */
   virtual void ioParam_nf(enum ParamsIOFlag ioFlag) override;
   /** @} */ // end of CloneLayerGeometry parameters

  public:
   CloneLayerGeometry(char const *name, PVParams *params, Communicator const *comm);
   virtual ~CloneLayerGeometry();

  protected:
   CloneLayerGeometry();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   // data members of CloneLayerGeometry

}; // class CloneLayerGeometry

} // namespace PV

#endif // CLONELAYERGEOMETRY_HPP_
