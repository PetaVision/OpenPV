/*
 * WeightsFile.hpp
 *
 *  Created on: Jan 19, 2022
 *      Author: peteschultz
 */

#ifndef WEIGHTSFILE_HPP_
#define WEIGHTSFILE_HPP_

#include "structures/WeightData.hpp"

#include <memory>

namespace PV {

/**
 * A class that provides a common interface for LocalPatchWeightsFile and SharedWeightsFile
 */
class WeightsFile : public CheckpointerDataInterface{
  protected:
   WeightsFile(std::shared_ptr<WeightData> weightData) : CheckpointerDataInterface() {
      mWeightData = weightData;
   }
   WeightsFile() = delete;
   ~WeightsFile() {}

  public:
   virtual void read() = 0;
   virtual void read(double &timestamp) = 0;
   virtual void write(double timestamp) = 0;

   virtual void truncate(int index) = 0;

   int getIndex() const { return mIndex; }
   virtual void setIndex(int index) { mIndex = index; }

  protected:
   std::shared_ptr<WeightData> mWeightData = nullptr;

  private:
   int mIndex = 0;

}; // class WeightsFile

} // namespacePV

#endif // WEIGHTSFILE_HPP_
