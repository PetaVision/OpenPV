/*
 * LayerGeometry.hpp
 *
 *  Created on: Apr 6, 2018
 *      Author: pschultz
 */

#ifndef LAYERGEOMETRY_HPP_
#define LAYERGEOMETRY_HPP_

#include "columns/BaseObject.hpp"
#include "include/PVLayerLoc.h"
#include <vector>

namespace PV {

/**
 * A component, used by HyPerLayer, that reads and writes
 * the parameters nxScale, nyScale, and nf; and creates a PVLayerLoc
 * based on those parameter values.
 */
class LayerGeometry : public BaseObject {
  protected:
   /**
    * List of parameters needed from the LayerGeometry class
    * @name LayerGeometry Parameters
    * @{
    */

   /**
    * @brief nxScale: Defines the relationship between the x column size and the layer size.
    * @details Must be 2^n or 1/2^n
    */
   virtual void ioParam_nxScale(enum ParamsIOFlag ioFlag);

   /**
    * @brief nyScale: Defines the relationship between the y column size and the layer size.
    * @details Must be 2^n or 1/2^n
    */
   virtual void ioParam_nyScale(enum ParamsIOFlag ioFlag);

   /**
    * @brief nf: Defines the number of features the layer has
    */
   virtual void ioParam_nf(enum ParamsIOFlag ioFlag);
   /** @} */ // end of LayerGeometry parameters

  public:
   LayerGeometry(char const *name, HyPerCol *hc);
   virtual ~LayerGeometry();

   /**
    * If axis is 'x', sets the PVLayerLoc's halo.lt and halo.rt to the
    * larger of their current value and the value of marginWidthNeeded.
    * If axis is 'y', does the same but for halo.dn and halo.up.
    * The return value is the resulting margin.
    */
   int requireMarginWidth(int marginWidthNeeded, char axis);

   static void synchronizeMarginWidths(LayerGeometry *geometry1, LayerGeometry *geometry2);

   void synchronizeMarginWidth(LayerGeometry *otherGeometry);

   /**
    * Returns the name of the connection's presynaptic layer.
    */
   PVLayerLoc const *getLayerLoc() const { return &mLayerLoc; }

   int getNumNeurons() const { return mNumNeurons; }
   int getNumExtended() const { return mNumExtended; }
   int getNumNeuronsAllBatches() const { return mNumNeuronsAllBatches; }
   int getNumExtendedAllBatches() const { return mNumExtendedAllBatches; }

   /**
    * Returns the scale factor of the layer geometry in the x-direction.
    * In terms of the input parameter nxScale, XScale = -log2(nxScale)
    * Thus, a positive XScale indicates that neighboring neurons are
    * farther apart than in a layer with nxScale = 0; and
    * negative indicates they are closer together.
    */
   int getXScale() const { return mXScale; }

   /**
    * Returns the scale factor of the layer geometry in the y-direction.
    * In terms of the input parameter nyScale, YScale = -log2(nyScale)
    * Thus, a positive YScale indicates that neighboring neurons are
    * farther apart than in a layer with nyScale = 0; and
    * negative indicates they are closer together.
    */
   int getYScale() const { return mYScale; }

  protected:
   LayerGeometry();

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * Calculates NumExtended and NumExtendedAllBatches from the fields of the LayerLoc data member.
    * Called when the size of the halo changes.
    */
   void updateNumExtended();

  private:
   /**
    * If a subclass needs to examine other objects, to get the values of nxScale, nyScale, or nf,
    * it can override this method, which LayerGeometry::communicateInitInfo calls
    * before filling in the fields of mLayerLoc.
    */
   virtual void communicateLayerGeometry(std::shared_ptr<CommunicateInitInfoMessage const> message);

   void setLayerLoc(PVLayerLoc *layerLoc);

  protected:
   float mNxScale   = 1.0;
   float mNyScale   = 1.0;
   int mNumFeatures = 1;

   PVLayerLoc mLayerLoc;
   int mNumNeurons            = 0;
   int mNumNeuronsAllBatches  = 0;
   int mNumExtended           = 0;
   int mNumExtendedAllBatches = 0;

   int mXScale = 1;
   int mYScale = 1;

   std::vector<LayerGeometry *> mSynchronizedMargins;

}; // class LayerGeometry

} // namespace PV

#endif // LAYERGEOMETRY_HPP_
