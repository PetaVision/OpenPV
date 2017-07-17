/*
 * InitVFromFile.hpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#ifndef INITVFROMFILE_HPP_
#define INITVFROMFILE_HPP_

#include "BaseInitV.hpp"
#include "utils/BufferUtilsPvp.hpp"

namespace PV {

class InitVFromFile : public BaseInitV {
  protected:
   /**
    * List of parameters needed from the InitVFromFile class
    * @name InitVFromFile Parameters
    * @{
    */

   /**
    * @brief VFilename: The path to the file with the initial values.
    * Relative paths are relative to the working directory.
    */
   virtual void ioParam_Vfilename(enum ParamsIOFlag ioFlag);

   /**
    * @brief frameNumber: selects which frame of the pvp file to use.
    * The default value is zero. Note that this parameter is zero-indexed:
    * for example, if a pvp file has five frames, the allowable values of this
    * parameter are 0 through 4, inclusive.
    * When batching, the batch element 0 loads the indicated frame,
    * batch element 1 loads frameNumber + 1, etc. If the number of frames in
    * the file is exhausted, the file wraps around to the beginning.
    */
   virtual void ioParam_frameNumber(enum ParamsIOFlag ioFlag);
   /** @} */
  public:
   InitVFromFile(char const *name, HyPerCol *hc);
   virtual ~InitVFromFile();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual int calcV(float *V, PVLayerLoc const *loc) override;

  protected:
   InitVFromFile();
   int initialize(char const *name, HyPerCol *hc);
   void readDenseActivityPvp(
         float *V,
         PVLayerLoc const *loc,
         FileStream &fileStream,
         BufferUtils::ActivityHeader const &header);

  private:
   int initialize_base();

  private:
   char *mVfilename = nullptr;
   int mFrameNumber = 0;
}; // end class InitVFromFile

} // end namespace PV

#endif /* INITVFROMFILE_HPP_ */
