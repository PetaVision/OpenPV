/*
 * PatchGeometry.hpp
 *
 *  Created on: Jul 21, 2017
 *      Author: Pete Schultz
 */

#ifndef PATCHGEOMETRY_HPP_
#define PATCHGEOMETRY_HPP_

#include "components/Patch.hpp"
#include "include/PVLayerLoc.h"
#include <string>
#include <vector>

namespace PV {

/**
 * PatchGeometry encapsulates the patch geometry of a HyPerConn.
 *
 * Instantiate with the size of the patch in postsynaptic space that is affected by one presynaptic
 * neuron, and the pre- and post-synaptic PVLayerLocs.
 * Then call the allocateDataStructures() method.
 *
 * The PatchGeometry object defines a connectivity betwee the pre- and post- synaptic layers.
 * Each pre-synaptic neuron maps to a 3D block in the postsynaptic layer, whose dimensions
 * are specified by the patchSizeX, patchSizeY, and patchSizeF arguments of the constructor.
 * However, the full 3D block is not always generated, because the block may extend off
 * the edge of the restricted post-synaptic layer.
 *
 * There is a patch for each pre-synaptic neuron in extended space, and the patches are indexed
 * the same way as a layer's neurons are indexed: feature index spins fastest, then the
 * x-dimension index, and finally the y-dimension index spins slowest. The patch tries to influence
 * a PatchSizeX-by-PatchSizeY-by-PatchSizeF region of the restricted postsynaptic layer, but
 * because of the edge of the layer, the resulting region may be smaller than this.
 *
 * The getPatch(index) method returns information on the active part of the patch.
 * The nx and ny fields give the dimensions of this portion; and the offset field gives the index
 * of the start of this portion into a PatchSizeX-by-PatchSizeY-by-PatchSizeF block of memory,
 * indexed in the usual PetaVision way.
 *
 * The getGSynPatchStart(index) method returns the index, in restricted postsynaptic space, of
 * the start of the active region.
 *
 * The getAPostOffset(index) method returns the extended postsynaptic index of the neuron whose
 * restricted index is GSynPatchStart.
 *
 * Note that the PatchGeometry object defines which pre- and post-synaptic neurons are connected,
 * but does not provide any data structures for defining the strengths of the connections.
 * For HyPerConn (both shared and nonshared weights), these structures are provided by the
 * Weights class.
 */
class PatchGeometry {

  public:
   /**
    * The constructor for the PatchGeometry object. Note that it sets member variables, but
    * does not allocate the data structures used by getPatch(), getGSynPatchStart(), or
    * getAPostOffset. Note that it is necessary to call the public method allocateDataStructures()
    * to complete the initialization.
    *
    * The constructor makes copies of the pre- and post-synaptic PVLayerLoc objects. It also records
    * whether the pointers to preLoc and postLoc point to the same memory location:
    * see getSelfConnectionFlag().
    */
   PatchGeometry(
         std::string const &name,
         int patchSizeX,
         int patchSizeY,
         int patchSizeF,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc);

   /** The destructor for the PatchGeometry object. */
   ~PatchGeometry() {}

   /**
    * Copies the given halos into the halos of the presynaptic and postsynaptic PVLayerLoc objects
    * It is an error to call this method after allocateDataStructures method.
    */
   void setMargins(PVHalo const &preHalo, PVHalo const &postHalo);

   /**
    * Allocates the vectors for the patch data, GSynPatchStart, and APostOffset structures.
    * The PatchGeometry object is not completely initialized until this method is called.
    * Once allocateDataStructures() is called once, subsequent calls return immediately
    * and have no effect.
    */
   void allocateDataStructures();

   /**
    * get-method for PatchSizeX, the size in the x-direction of the patch from one pre-synaptic
    * neuron into post-synaptic space.
    */
   int getPatchSizeX() const { return mPatchSizeX; }

   /**
    * get-method for PatchSizeY, the size in the y-direction of the patch from one pre-synaptic
    * neuron into post-synaptic space.
    */
   int getPatchSizeY() const { return mPatchSizeY; }

   /**
    * get-method for PatchSizeF, the number of features in post-synaptic space that are affected
    * by one pre-synaptic neuron. Currently, PatchSizeF is always equal to PostLoc.nf.
    * That is, in the feature dimension the connection is all-to-all.
    */
   int getPatchSizeF() const { return mPatchSizeF; }

   /**
    * Returns getPatchSizeX() * getPatchSizeY() * getPatchSizeF(),
    * the overall number of items in a patch.
    */
   int getPatchSizeOverall() const { return mPatchSizeX * mPatchSizeY * mPatchSizeF; }

   /** get-method to retrieve a constant reference to the pre-synaptic PVLayerLoc. */
   PVLayerLoc const &getPreLoc() const { return mPreLoc; }

   /** get-method to retrieve a constant reference to the pre-synaptic PVLayerLoc. */
   PVLayerLoc const &getPostLoc() const { return mPostLoc; }

   /**
    * Returns the number of patches in the x-direction. This quantity is equal to
    * getPreLoc().nx + getPreLoc().halo.lt + getPreLoc().halo.rt
    */
   int getNumPatchesX() const { return mNumPatchesX; }

   /**
    * Returns the number of patches in the y-direction. This quantity is equal to
    * getPreLoc().ny + getPreLoc().halo.dn + getPreLoc().halo.up
    */
   int getNumPatchesY() const { return mNumPatchesY; }

   /**
    * Returns the number of patches in the feature direction. This quantity is equal to
    * getPreLoc().nf
    */
   int getNumPatchesF() const { return mNumPatchesF; }

   /** Returns the overall number of patches in the patch geometry */
   int getNumPatches() const { return getNumPatchesX() * getNumPatchesY() * getNumPatchesF(); }

   /**
    * Returns the number of kernels in the x-direction. This quantity is equal to
    * getPreLoc().nx / getPostLoc().nx if that quotient is greater than 1; 1 otherwise.
    */
   int getNumKernelsX() const { return mNumKernelsX; }

   /**
    * Returns the number of patches in the y-direction. This quantity is equal to
    * getPreLoc().ny / getPostLoc().ny if that quotient is greater than 1; 1 otherwise.
    */
   int getNumKernelsY() const { return mNumKernelsY; }

   /**
    * Returns the number of kernels in the feature direction. This quantity is equal to
    * getPreLoc().nf
    */
   int getNumKernelsF() const { return mNumKernelsF; }

   /** Returns the overall number of patches in the patch geometry */
   int getNumKernels() const { return getNumKernelsX() * getNumKernelsY() * getNumKernelsF(); }

   /** Returns a nonmutable reference to the patch info for the given patch index. */
   Patch const &getPatch(int patchIndex) const { return mPatchVector[patchIndex]; }

   /** Returns the GSynPatchStart value for the indicated patch index */
   std::size_t getGSynPatchStart(int patchIndex) const { return mGSynPatchStart[patchIndex]; }

   /** Returns a nonmutable reference to the vector of GSynPatchStart values. */
   std::vector<std::size_t> const &getGSynPatchStart() const { return mGSynPatchStart; }

   /** Returns the APostOffset value for the indicated patch index */
   std::size_t getAPostOffset(int patchIndex) const { return mAPostOffset[patchIndex]; }

   /** Returns the UnshrunkenStart value for the indicated patch index */
   long getUnshrunkenStart(int patchIndex) const { return mUnshrunkenStart[patchIndex]; }

   /** Returns the item index of the postsynaptic-perspective patch corresponding to the
     * the given item index of the presynaptic-perspective patch with the given kernel index.
     */
   std::size_t getTransposeItemIndex(int kernelIndex, int itemInPatch) const {
      return mTransposeItemIndex[kernelIndex][itemInPatch];
   }

   /** Returns a nonmutable reference to the vector of APostOffset values. */
   std::vector<std::size_t> const &getAPostOffset() const { return mAPostOffset; }

   int getPatchStrideX() const { return mPatchStrideX; }
   int getPatchStrideY() const { return mPatchStrideY; }
   int getPatchStrideF() const { return mPatchStrideF; }

   /**
    * Returns log_2(preLoc.nx / postLoc.nx), using floating-point division.
    * The PatchGeometry class requires that this quantity be an integer.
    */
   int getLog2ScaleDiffX() const { return mLog2ScaleDiffX; }

   /**
    * Returns log_2(preLoc.ny / postLoc.ny), using floating-point division.
    * The PatchGeometry class requires that this quantity be an integer.
    */
   int getLog2ScaleDiffY() const { return mLog2ScaleDiffY; }

   /**
    * Returns true if the PatchGeometry object was instantiated with pre- and post-
    * synaptic PVLayerLoc pointing to the same memory location.  Returns false otherwise, even if
    * the values of the PVLayerLoc fields are all equal.
    */
   bool getSelfConnectionFlag() const { return mSelfConnectionFlag; }

   static int calcPatchStartInPost(
         int indexRestrictedPre,
         int patchSize,
         int numNeuronsPre,
         int numNeuronsPost);

  private:
   /** Called internally by the constructor */
   void initialize(
         std::string const &name,
         int patchSizeX,
         int patchSizeY,
         int patchSizeF,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc);

   /**
    * Checks that the given pre- and post-synaptic dimensions are multiples of powers of two
    * of each other, and that the patch size is consistent with the quotient.
    * Returns log2(numPostRestricted/numPreRestricted)
    */
   static int verifyPatchSize(int numPreRestricted, int numPostRestricted, int patchSize);

   /**
    * Checks the patch size in the x- and y-directions, and that the PatchSizeF argument
    * agrees with the postLoc.nf argument.
    */
   void verifyPatchSize();

   /**
    * Called internally by allocateDataStructures, to compute the vectors of Patch objects,
    * GSynPatchStart values, and APostOffset values.
    */
   void setPatchGeometry();

   /**
    * Called internally by allocateDataStructures, to compute the TransposeItemIndex vectors.
    */
   void setTransposeItemIndices();

   static void calcPatchData(
         int index,
         int numPreRestricted,
         int preStartBorder,
         int preEndBorder,
         int numPostRestricted,
         int postStartBorder,
         int postEndBorder,
         int patchSize,
         int *patchDim,
         int *patchStart,
         int *postPatchStartRestricted,
         int *postPatchStartExtended,
         int *postPatchUnshrunkenStart);

  private:
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   PVLayerLoc mPreLoc;
   PVLayerLoc mPostLoc;
   int mNumPatchesX;
   int mNumPatchesY;
   int mNumPatchesF;
   int mNumKernelsX;
   int mNumKernelsY;
   int mNumKernelsF;

   std::vector<Patch> mPatchVector;
   std::vector<std::size_t> mGSynPatchStart;
   std::vector<std::size_t> mAPostOffset;
   std::vector<long> mUnshrunkenStart;
   std::vector<std::vector<int>> mTransposeItemIndex;

   int mPatchStrideX;
   int mPatchStrideY;
   int mPatchStrideF;

   int mLog2ScaleDiffX;
   int mLog2ScaleDiffY;
   bool mSelfConnectionFlag; // True if instantiated with preLoc and postLoc pointers equal
}; // end class PatchGeometry

} // end namespace PV

#endif // PATCHGEOMETRY_HPP_
