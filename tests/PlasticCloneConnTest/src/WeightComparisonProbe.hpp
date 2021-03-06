/*
 * WeightComparisonProbe.hpp
 *
 *  Created on: Mar 26, 2017
 *      Author: pschultz
 */

#ifndef WEIGHTCOMPARISONPROBE_HPP_
#define WEIGHTCOMPARISONPROBE_HPP_

#include <columns/ComponentBasedObject.hpp>
#include <probes/ColProbe.hpp>

#include <vector>

namespace PV {

/**
 * A probe to verify that all connections in the column have the same weights.
 * This probe is used by PlasticCloneConnTest to verify that the results
 * are the same no matter which connection is the original and which is
 * the probe.
 */
class WeightComparisonProbe : public PV::ColProbe {
  public:
   /**
    * Public constructor for the ColProbe class.
    */
   WeightComparisonProbe(char const *name, PV::PVParams *params, PV::Communicator const *comm);

   /**
    * Destructor for the ColProbe class.
    */
   virtual ~WeightComparisonProbe();

  protected:
   /**
    */
   void initialize(char const *name, PV::PVParams *params, PV::Communicator const *comm);
   /**
    * Assembles the list of HyPerConns in the column.
    */
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * Sets the number of arbors and the number of weights in each arbor,
    * and verifies that these quantities are the same for each connection
    * in the column.
    */
   virtual Response::Status allocateDataStructures() override;

   virtual bool needRecalc(double timevalue) override { return true; }
   virtual double referenceUpdateTime(double simTime) const override;
   virtual void calcValues(double timevalue) override {}
   /**
    * Exits with an error if any connections are found to be different
    * from each other.
    */
   virtual Response::Status outputState(double simTime, double deltaTime) override;

  private:
   int initialize_base();

  private:
   std::vector<PV::ComponentBasedObject *> mConnectionList;
   int mNumArbors;
   int mNumWeightsInArbor;
}; // end class WeightComparisonProbe
} // end namespace PV

#endif // WEIGHTCOMPARISONPROBE_HPP_
