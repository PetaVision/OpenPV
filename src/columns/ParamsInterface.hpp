/*
 * ParamsInterface.hpp
 *
 *  Created on May 16, 2018
 *      Author: Pete Schultz
 */

#ifndef PARAMSINTERFACE_HPP_
#define PARAMSINTERFACE_HPP_

#include "checkpointing/CheckpointerDataInterface.hpp"
#include "io/PVParams.hpp"

namespace PV {

/**
 * ParamsInterface derives from CheckpointDataInterface, and adds a standard interface
 * for reading from a PVParams database and writing to params files (either .params or .lua).
 */
class ParamsInterface : public CheckpointerDataInterface {
   // Function members
  public:
   /**
    * A method that reads the parameters for the group whose name matches the name of the object.
    * It, along with writeParams(), is a wrapper around ioParams, so that readParams and
    * writeParams automatically run through the same parameters in the same order.
    */
   void readParams() { ioParams(PARAMS_IO_READ, false, false); }

   /**
    * A method that writes the parameters for the group whose name matches the name of the object.
    * It, along with readParams(), is a wrapper around ioParams, so that readParams and writeParams
    * automatically run through the same parameters in the same order.
    */
   void writeParams() { ioParams(PARAMS_IO_WRITE, true, true); }

   /**
    * Method for reading or writing the params from group in the parent HyPerCol's parameters.
    * The group from params is selected using the name of the connection.
    *
    * If ioFlag is set to write, the printHeader and printFooter flags control whether
    * a header and footer for the parameter group is produces. These flags are set to true
    * for layers, connections, and probes; and set to false for weight initializers and
    * normalizers. If ioFlag is set to read, the printHeader and printFooter flags are ignored.
    *
    * Note that ioParams is not virtual.  To add parameters in a derived class, override
    * ioParamsFillGroup.
    */
   void ioParams(enum ParamsIOFlag ioFlag, bool printHeader, bool printFooter);

   char const *getName() const { return mName; }
   PVParams *parameters() const { return mParams; } // TODO: change to getParams()
   std::string const &getObjectType() const { return mObjectType; }

  protected:
   int initialize(char const *name, PVParams *params);
   void setName(char const *name);
   void setParams(PVParams *params);
   virtual void setObjectType();
   void ioParamsStartGroup(enum ParamsIOFlag ioFlag);

   /**
    * The virtual method for reading parameters from the PVParams database, and writing
    * to the output params file.
    *
    * Derived classes with additional parameters typically override ioParamsFillGroup to call the
    * base class's ioParamsFillGroup
    * method and then call ioParam_[parametername] for each of their parameters.
    * The ioParam_[parametername] methods usually calls the PVParams object's ioParamValue() and
    * related methods, to ensure that all parameters that get read also get written to the
    * outputParams-generated file.
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) { return PV_SUCCESS; }
   void ioParamsFinishGroup(enum ParamsIOFlag);

  private:
   // Data members
  protected:
   char *mName       = nullptr;
   PVParams *mParams = nullptr;
   std::string mObjectType;

  private:
}; // end class ParamsInterface

} // end namespace PV

#endif // PARAMSINTERFACE_HPP_
