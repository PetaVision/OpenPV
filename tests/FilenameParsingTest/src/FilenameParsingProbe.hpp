/*
 * LayerProbe.h
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#ifndef FILENAMEPARSINGPROBE_HPP_
#define FILENAMEPARSINGPROBE_HPP_

#include <layers/FilenameParsingGroundTruthLayer.hpp>
#include <probes/LayerProbe.hpp>

/**
 * The base class for probes attached to layers.
 */
class FilenameParsingProbe : public PV::LayerProbe {

   // Methods
  public:
   FilenameParsingProbe(const char *name, PV::HyPerCol *hc);
   virtual ~FilenameParsingProbe();

  protected:
   FilenameParsingProbe();
   int initialize(const char *name, PV::HyPerCol *hc);
   virtual int
   communicateInitInfo(std::shared_ptr<PV::CommunicateInitInfoMessage const> message) override;
   virtual int calcValues(double timevalue) override { return 0; }
   virtual int outputState(double timestamp) override;

  private:
   int initialize_base();

  private:
   PV::FilenameParsingGroundTruthLayer *mFilenameParsingLayer = nullptr;
   int mInputDisplayPeriod                                    = 0;

   // This vector gives the category corresponding to each line of
   // InputImages.txt. For example, the first line of InputImages.txt is
   // automobile1.png; and automobile is (zero-indexed) line number 1
   // of Classes.txt. Hence the first element of mCategories is 1.
   // The second line of InputImages.txt is deer1.png; and deer is
   // (zero-indexed) line number 4 of Classes.txt. Hence the second
   // element of mCategories is 4, etc.
   //
   // If either Classes.txt or InputImages.txt changes, this vector
   // will need to be changed to match.
   std::vector<int> const mCategories = {1, 4, 9, 9, 8, 3, 5, 0, 7, 4,
                                         1, 8, 2, 2, 5, 3, 6, 0, 6, 7};
};

#endif /* FILENAMEPARSINGPROBE_HPP_ */
