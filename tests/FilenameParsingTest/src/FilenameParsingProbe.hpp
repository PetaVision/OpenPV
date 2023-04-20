/*
 * LegacyLayerProbe.h
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#ifndef FILENAMEPARSINGPROBE_HPP_
#define FILENAMEPARSINGPROBE_HPP_

#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/Messages.hpp>
#include <io/PVParams.hpp>
#include <layers/FilenameParsingLayer.hpp>
#include <observerpattern/Response.hpp>
#include <probes/TargetLayerComponent.hpp>

#include <memory>
#include <vector>

using namespace PV;

/**
 * The base class for probes attached to layers.
 */
class FilenameParsingProbe : public BaseObject {

   // Methods
  public:
   FilenameParsingProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~FilenameParsingProbe();

  protected:
   FilenameParsingProbe() {}
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void initMessageActionMap() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   Response::Status outputState(std::shared_ptr<LayerOutputStateMessage const> message);

   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   Response::Status respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message);

  private:
   int mInputDisplayPeriod = 0;

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

   std::shared_ptr<TargetLayerComponent> mProbeTargetLayerLocator = nullptr;
   FilenameParsingLayer *mFilenameParsingLayer                    = nullptr;
};

#endif /* FILENAMEPARSINGPROBE_HPP_ */
