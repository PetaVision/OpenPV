#ifndef CONNECTIONSPECS_HPP_
#define CONNECTIONSPECS_HPP_

namespace PV {

/**
 * A simple class used by LocalPatchWeightsFileTest to hold paramabers to define a connection.
 * It has no behavior other than a constructor and accessor methods.
 */
struct ConnectionSpecs {
  public:
   ConnectionSpecs(
         int numArbors,
         int patchSizeX,
         int patchSizeY,
         int patchSizeF,
         int nxGlobalRestrictedPre,
         int nyGlobalRestrictedPre,
         int nfPre,
         int nxGlobalRestrictedPost,
         int nyGlobalRestrictedPost) :
      mNumArbors(numArbors),
      mPatchSizeX(patchSizeX),
      mPatchSizeY(patchSizeY),
      mPatchSizeF(patchSizeF),
      mNxGlobalRestrictedPre(nxGlobalRestrictedPre),
      mNyGlobalRestrictedPre(nyGlobalRestrictedPre),
      mNfPre(nfPre),
      mNxGlobalRestrictedPost(nxGlobalRestrictedPost),
      mNyGlobalRestrictedPost(nyGlobalRestrictedPost) {}
   ConnectionSpecs() = delete;
   ~ConnectionSpecs() {}
      
   int getNumArbors() const { return mNumArbors; }
   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   int getNxGlobalRestrictedPre() const { return mNxGlobalRestrictedPre; }
   int getNyGlobalRestrictedPre() const { return mNyGlobalRestrictedPre; }
   int getNfPre() const { return mNfPre; }
   int getNxGlobalRestrictedPost() const { return mNxGlobalRestrictedPost; }
   int getNyGlobalRestrictedPost() const { return mNyGlobalRestrictedPost; }

  private:
   int mNumArbors;
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   int mNxGlobalRestrictedPre;
   int mNyGlobalRestrictedPre;
   int mNfPre;
   int mNxGlobalRestrictedPost;
   int mNyGlobalRestrictedPost;
};

} // namespace PV

#endif // CONNECTIONSPECS_HPP_
