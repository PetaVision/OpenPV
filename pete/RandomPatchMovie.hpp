/*
 * RandomPatchMovie.hpp
 *
 *      Author: pschultz
 */

#ifndef RANDOMPATCHMOVIE_HPP_
#define RANDOMPATCHMOVIE_HPP_

#include <src/layers/Movie.hpp>
#include <src/utils/pv_random.h>
#include <src/io/imageio.hpp>
#include <limits.h>

namespace PV {

class RandomPatchMovie : public Image {
public:
   RandomPatchMovie(const char * name, HyPerCol * hc);
   RandomPatchMovie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float defaultDisplayPeriod = DISPLAY_PERIOD);
   virtual ~RandomPatchMovie();
   virtual int updateState(float time, float dt);
   virtual int outputState(float time, bool last=false);

   float getDisplayPeriod() { return displayPeriod; }

   int getFileIndex() {return fileIndex; }
   virtual int getRandomOffsets(const PVLayerLoc * imgloc, int * offsetXptr, int * offsetYptr);
   virtual int getRandomFileIndex();

protected:
   int initialize_base();
   int initializeRandomPatchMovie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float defaultDisplayPeriod);
   int readOffsets(PVParams * params);
   int retrieveRandomPatch();
   virtual bool updateImage(float time, float dt);
   char * getRandomFilename();

   float displayPeriod;
   float nextDisplayTime;
   int numImageFiles;
   int * imageFilenameIndices;
   char * listOfImageFiles;
   int fileIndex;
}; // end class RandomPatchMovie

}  // end namespace PV

#endif /* RANDOMPATCHMOVIE_HPP_ */
