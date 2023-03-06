#include <arch/mpi/mpi.h>
#include <columns/HyPerCol.hpp>
#include <columns/Messages.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <io/PVParams.hpp>
#include <observerpattern/ObserverTable.hpp>
#include <probes/ColumnEnergyProbe.hpp>
#include <probes/EnergyProbeComponent.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>
#include <memory>
#include <string>

using PV::HyPerCol;
using PV::ColumnEnergyProbe;
using PV::PV_Init;
using PV::PVParams;
using PV::EnergyProbeComponent;

EnergyProbeComponent initEnergyProbeObject(
      HyPerCol &hypercol,
      std::string const &componentName,
      std::string const &energyProbeName,
      double coefficient);
PVParams generateParams(std::string const &componentName, std::string const &energyProbeName, double coefficient, MPI_Comm mpiComm);
int run(PV::PV_Init &pv_init);

int main(int argc, char **argv) {
   PV_Init pv_init(&argc, &argv, false);
   int status = run(pv_init);

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

PVParams generateParams(std::string const &componentName, std::string const &energyProbeName, double coefficient, MPI_Comm mpiComm) {
   std::string paramsString;
   paramsString.append("debugParsing = false;\n");
   paramsString.append("EnergyProbeComponent \"").append(componentName).append("\" = {\n");
   paramsString.append("   energyProbe = \"").append(energyProbeName).append("\";\n");
   paramsString.append("   coefficient = \"").append(std::to_string(coefficient)).append("\";\n");
   paramsString.append("};\n");

   PVParams params(paramsString.data(), paramsString.size(), 1UL, mpiComm);
   return params;
}

EnergyProbeComponent initEnergyProbeObject(
      HyPerCol &hypercol,
      std::string const &componentName,
      std::string const &energyProbeName,
      double coefficient) {
   std::string paramsString;

   MPI_Comm mpiComm = hypercol.getCommunicator()->globalCommunicator();
   PVParams params = generateParams(componentName, energyProbeName, coefficient, mpiComm);

   EnergyProbeComponent energyProbeObject(componentName.c_str(), &params);
   energyProbeObject.ioParamsFillGroup(PV::PARAMS_IO_READ);

   PV::ObserverTable objectTable = hypercol.getAllObjectsFlat();
   auto communicateMessage       = std::make_shared<PV::CommunicateInitInfoMessage>(
         &objectTable,
         hypercol.getDeltaTime(),
         hypercol.getNxGlobal(),
         hypercol.getNyGlobal(),
         hypercol.getNBatchGlobal(),
         hypercol.getNumThreads());
   energyProbeObject.communicateInitInfo(communicateMessage);

   return energyProbeObject;
}

int run(PV::PV_Init &pv_init) {
   PV::HyPerCol hypercol(&pv_init);
   hypercol.allocateColumn();

   std::string componentName("Probe");
   std::string columnEnergyProbeName("TestColumnEnergyProbe");
   double coefficient = 2.25;
   EnergyProbeComponent energyComponentProbeObj =
         initEnergyProbeObject(hypercol, componentName, columnEnergyProbeName, coefficient);

   auto *objectFromColumn = hypercol.getObjectFromName(columnEnergyProbeName);
   ColumnEnergyProbe *probeFromColumn = dynamic_cast<ColumnEnergyProbe *>(objectFromColumn);
   ColumnEnergyProbe *probeFromEnergyProbeComponent = energyComponentProbeObj.getEnergyProbe();
   FatalIf(
         probeFromEnergyProbeComponent != probeFromColumn,
         "EnergyProbeComponent::getEnergyProbe() failed (return value %p instead of %p)\n",
         probeFromEnergyProbeComponent,
         probeFromColumn);
   return PV_SUCCESS;
}
