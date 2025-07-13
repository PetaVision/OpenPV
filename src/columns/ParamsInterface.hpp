/*
 * ParamsInterface.hpp
 *
 *  Created on May 16, 2018
 *      Author: Pete Schultz
 */

#ifndef PARAMSINTERFACE_HPP_
#define PARAMSINTERFACE_HPP_

#include "checkpointing/CheckpointerDataInterface.hpp"
#include "params/ParamGroup.hpp"
#include "params/ParamsIO.hpp"
#include <memory>

namespace PV {

/**
 * ParamsInterface derives from CheckpointDataInterface, and adds a standard interface
 * for reading from a PVParams database and writing to params files (either .params or .lua).
 */
class ParamsInterface : public CheckpointerDataInterface {
   // Function members
  protected:
   /**
    * @brief initializeFromCheckpointFlag: If set to true, initialize using the checkpoint directory
    * set in HyPerCol.
    * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
    * Unlike most params file params, this flag is read by every ParamsInterface object, including
    * components within an object with the same ParameterGroup. The flag will be written to the
    * output params file only if the Boolean data member mWriteInitializeFromCheckpointFlag is true.
    * Derived classes should be written so that initializeFromCheckpointFlag is only written once
    * per parameter group. Currently, only HyPerLayer and BaseConnection set the flag.
    */
   void ioParam_initializeFromCheckpointFlag(ParamsIOSwitch ioSwitch);

  public:
   virtual ~ParamsInterface();

   /**
    * Method for reading or writing the params from group in the parent HyPerCol's parameters.
    * The group from params is selected using the name of the connection.
    *
    * If ioSwitch is set to write, the printHeader and printFooter flags control whether
    * a header and footer for the parameter group is produces. These flags are set to true
    * for layers, connections, and probes; and set to false for weight initializers and
    * normalizers. If ioSwitch is set to read, the printHeader and printFooter flags are ignored.
    *
    * Note that ioParams is not virtual.  To add parameters in a derived class, override
    * ioParamsFillGroup.
    */
   void ioParams(ParamsIOSwitch ioSwitch, bool printHeader, bool printFooter);

   char const *getName() const { return mName.c_str(); }
   char const *getKeyword() const { return mKeyword.c_str(); }
   std::shared_ptr<ParamsIO> getParamsIO() { return mParamsIO; }
   std::string const &getObjectType() const { return mObjectType; }

  protected:
   int initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   void setName(std::string const &name);
   void setKeyword(std::string const &keyword);
   void setParams(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual void setObjectType();
   void ioParamsStartGroup(ParamsIOSwitch ioSwitch);

   /**
    * The virtual method for reading parameters from the PVParams database, and writing
    * to the output params file.
    *
    * The base class ioParamsFillGroup handles the Boolean parameter initializeFromCheckpointFlag.
    *
    * Derived classes with additional parameters typically override ioParamsFillGroup to call the
    * base class's ioParamsFillGroup
    * method and then call ioParam_[parametername] for each of their parameters. The
    * ioParam_[parametername] methods usually calls the PVParams object's ioParam() method, to
    * ensure that all parameters that get read also get written to the outputParams-generated file.
    */
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) { return PV_SUCCESS; }

   void ioParamsFinishGroup(ParamsIOSwitch ioSwitch);

   // Data members
  protected:
   std::string mName;
   std::string mKeyword;
   std::shared_ptr<ParamsIO> mParamsIO;
   std::string mObjectType;

   // A flag for whether ioParams() writes initializeFromCheckpointFlag to the output params file.
   bool mWriteInitializeFromCheckpointFlag = false;

  private:
}; // end class ParamsInterface

} // end namespace PV

#endif // PARAMSINTERFACE_HPP_
