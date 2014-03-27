/*
 * RandomPatchMovie.hpp
 *
 *      Author: pschultz
 */

#ifndef RANDOMPATCHMOVIE_HPP_
#define RANDOMPATCHMOVIE_HPP_

#include <layers/Movie.hpp>
#include <io/imageio.hpp>
#include <limits.h>

namespace PV {

class RandomPatchMovie : public Image {
public:
   RandomPatchMovie(const char * name, HyPerCol * hc);
   virtual ~RandomPatchMovie();
   virtual int updateState(double timed, double dt);
   virtual int outputState(double timed, bool last=false);

   float getDisplayPeriod() { return displayPeriod; }

   int getFileIndex() {return fileIndex; }
   virtual int getRandomOffsets(const PVLayerLoc * imgloc, int * offsetXptr, int * offsetYptr);
   virtual int getRandomFileIndex();

protected:
   RandomPatchMovie();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imagePath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imageListPath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);
   int readOffsets();
   int retrieveRandomPatch();
   virtual bool updateImage(double timed, double dt);
   char * getRandomFilename();

   float displayPeriod;
   float nextDisplayTime;
   int numImageFiles;
   int * imageFilenameIndices;
   char * imageListPath;
   char * listOfImageFiles;
   int fileIndex;
   const char * patchposfilename;
   FILE * patchposfile;
   float skipLowContrastPatchProb;
   float lowContrastThreshold;
   uint4 rng;

private:
   int initialize_base();
}; // end class RandomPatchMovie

}  // end namespace PV

#endif /* RANDOMPATCHMOVIE_HPP_ */
