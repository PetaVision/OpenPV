/*
 * WeightsFile.hpp
 *
 *  Created on: Jan 19, 2022
 *      Author: peteschultz
 */

#ifndef WEIGHTSFILE_HPP_
#define WEIGHTSFILE_HPP_

#include "structures/WeightData.hpp"

namespace PV {

/**
 * A class that provides a common interface for LocalPatchWeightsFile and SharedWeightsFile
 */
class WeightsFile : public CheckpointerDataInterface{
  protected:
   WeightsFile() : CheckpointerDataInterface() {}
   ~WeightsFile() {}

  public:
   virtual void read(WeightData &weightData) = 0;
   virtual void read(WeightData &weightData, double &timestamp) = 0;
   virtual void write(WeightData const &weightData, double timestamp) = 0;

   virtual void truncate(int index) = 0;

   int getIndex() const { return mIndex; }
   virtual void setIndex(int index) { mIndex = index; }

  private:
   int mIndex = 0;

}; // class WeightsFile

} // namespacePV

#endif // WEIGHTSFILE_HPP_
