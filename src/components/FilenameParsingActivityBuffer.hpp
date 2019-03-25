/*
 * FilenameParsingActivityBuffer.hpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */
#ifndef FILENAMEPARSINGACTIVITYBUFFER_HPP_
#define FILENAMEPARSINGACTIVITYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"

#include "components/InputLayerNameParam.hpp"
#include "layers/InputLayer.hpp"
#include <string>

namespace PV {

class FilenameParsingActivityBuffer : public ActivityBuffer {
  protected:
   /**
    * List of protected paramters needed from FilenameParsingActivityBuffer
    * @name FilenameParsingActivityBuffer Paramters
    * @{
    */

   /**
    * @brief clasList: path to the .txt file that holds the list of imageListPath features
    * that will parse to different classifications
    * @details If this is not specified, the layer will attempt to use "classes.txt" in the output
    * directory. The identifying strings must be separated by a new line, and mutually exclusive.
    * In the case of CIFAR images, the pictures are organized in folders /0/ /1/ /2/ ... etc,
    * therefore those are the classifers
    * When an image is passed to the movie layer, the classification is parsed and a corresponding
    * neuron is activated to a value set by
    * gtClassTrueValue and the remaining neurons are set to the value set by gtClassFalseValue
    */

   virtual void ioParam_classList(enum ParamsIOFlag ioFlag);

   /**
    * @brief gtClassTrueValue: defines value to be set for the neuron that matches classes.txt
    * classifer
    * @details Default: 1
    */

   virtual void ioParam_gtClassTrueValue(enum ParamsIOFlag ioFlag);

   /**
    * @brief gtClassFalseValue: defines value to be set for the neurons that do not match the
    * classes.txt classifer
    * @details Default: -1
    */
   virtual void ioParam_gtClassFalseValue(enum ParamsIOFlag ioFlag);

   /** @} */

  public:
   FilenameParsingActivityBuffer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~FilenameParsingActivityBuffer();
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual void updateBufferCPU(double timef, double dt) override;

  private:
   std::vector<std::string> mClasses;
   char *mInputLayerName    = nullptr;
   char *mClassListFileName = nullptr;
   InputLayer *mInputLayer  = nullptr;
   float mGtClassTrueValue  = 1.0f;
   float mGtClassFalseValue = 0.0f;
}; // end class FlenameParsingActivityBuffer

} // end namespace PV

#endif // FILENAMEPARSINGACTIVITYBUFFER_HPP_
