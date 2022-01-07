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
   LayerGeometry(char const *name, PVParams *params, Communicator const *comm);
   virtual ~LayerGeometry();

   /**
    * If axis is 'x', sets the PVLayerLoc's halo.lt and halo.rt to the larger of their current
    * value and the value of marginWidthNeeded. If axis is 'y', does the same but for halo.dn and
    * halo.up.  If the margin is thereby increased, any other LayerGeometry that was earlier
    * synchronized through a call to synchronizeMarginWidth() or synchronizeMarginWidths() has its 
    * margin requireMarginWidth() method called as well.
    */
   void requireMarginWidth(int marginWidthNeeded, char axis);

   /**
    * Sets each margin of each geometry to the max of that margin between the two geometries.
    * Additionally registers each geometry with the other, so that going forward, increasing
    * a margin of one geometry will cause the corresponding margin of the other to increase to
    * the same amount.
    */
   static void synchronizeMarginWidths(LayerGeometry *geometry1, LayerGeometry *geometry2);

   /**
    * Sets each margin of this object and the LayerGeometry specified by the otherGeometry
    * argument to the max of that margin between the two geometries.
    * Additionally registers otherGeometry with current object, so that going forward, increasing
    * a margin for this object will cause the corresponding margin of the other object to increase
    * to the same amount.
    */
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


   /**
    * Sets nBatch, nx, ny, kb0, kx0, ky0 of the given layerLoc,
    * based on nBatchGlobal, nxGlobal, nyGlobal, and the MPI arrangement given in the communicator.
    * The label argument is used only in error messages, to identify the source of the layerLoc.
    */
   static int setLocalLayerLocFields(PVLayerLoc *layerLoc, Communicator const *icComm, std::string const &label);

  protected:
   LayerGeometry();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

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

   void
   setLayerLoc(PVLayerLoc *layerLoc, std::shared_ptr<CommunicateInitInfoMessage const> message);

   /**
    * Multiplies scaleFactor and baseSize, and rounds to an integer.
    * If the product of scaleFactor and baseSize is an integer to a tolerance of 0.0001,
    * the result is put in scaledSize and the function returns PV_SUCCESS.
    * Otherwise, an error message is printed, an error message is printed and scaledSize is not set.
    * A return value of PV_SUCCESS indicates success and the scaledSize was set; otherwise the 
    * return value is PV_FAILURE and scaledSize is not changed.
    */
   int calculateScaledSize(int *scaledSize, float scaleFactor, int baseSize, char axis);

   /**
    * Returns true if globalSize divided by numProcesses gives a nonzero remainder, false otherwise.
    * If there is a remainder, and printErr is true, the process prints an error message
    * (but not a fatal error) on the assumption that globalSize is one of the dimensions,
    * in pixels, of a layer, and that numProcesses is the number of MPI processes in that direction.
    * The argument label is used to identify the layer in the error message, and is not used
    * if there is no remainder or if printErr is false.
    *
    * In the motivating use case, printErr is true for the global root process and false for the
    * other processes.
    */
   static bool checkRemainder(
      int globalSize, int numProcesses, std::string axis, std::string const &label, bool printErr);

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
