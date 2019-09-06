#ifndef PYTHONBINDINGS_HPP_
#define PYTHONBINDINGS_HPP_

#define PYTHON_MODULE_NAME      PythonBindings 
#define PYTHON_MODULE_NAME_STR "PythonBindings"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <bindings/Commander.hpp>


namespace py = pybind11;

namespace PV {

/*
   The PythonBindings class provides a Python interface for the Commander 
   class to be used with Python types and structures.
*/

   // TODO: Wrap getLayerSparseActivity

class PythonBindings {
  public:
   PythonBindings(py::dict args, std::string params);
   ~PythonBindings();

  private:
   Commander *mCmd;
   
  public:
   void                begin();
   double              advance(unsigned int steps);
   void                finish();
   py::array_t<float>  getConnectionWeights(const char *connName); 
   void                setConnectionWeights(const char *connName, py::array_t<float> *data); 
   py::tuple           getLayerSparseActivity(const char *layerName);
   py::array_t<float>  getLayerActivity(const char *layerName); 
   py::array_t<float>  getLayerState(const char *layerName); 
   void                setLayerState(const char *layerName, py::array_t<float> *data);
   bool                isFinished();
   py::array_t<double> getProbeValues(const char *probeName);
   bool                isRoot();
   void                waitForCommands();
   double              getLastCheckpointTime();
   void                checkpoint();
};

void err(std::string e) {
   py::dict d;
   d["flush"] = true;
   py::print("python error: " + e, d);
}



} /* namespace PV */

PYBIND11_MODULE( PYTHON_MODULE_NAME, m ) {
   m.doc() = "Python bindings for OpenPV";

   py::class_<PV::PythonBindings>(m, "PetaVision")
      .def(py::init<py::dict, std::string>())
      .def("begin",                     &PV::PythonBindings::begin)
      .def("advance",                   &PV::PythonBindings::advance)
      .def("finish",                    &PV::PythonBindings::finish)
      .def("get_connection_weights",    &PV::PythonBindings::getConnectionWeights)
      .def("set_connection_weights",    &PV::PythonBindings::setConnectionWeights)
      .def("get_layer_sparse_activity", &PV::PythonBindings::getLayerSparseActivity)
      .def("get_layer_activity",        &PV::PythonBindings::getLayerActivity)
      .def("get_layer_state",           &PV::PythonBindings::getLayerState)
      .def("set_layer_state",           &PV::PythonBindings::setLayerState)
      .def("is_finished",               &PV::PythonBindings::isFinished)
      .def("get_probe_values",          &PV::PythonBindings::getProbeValues)
      .def("is_root",                   &PV::PythonBindings::isRoot)
      .def("wait_for_commands",         &PV::PythonBindings::waitForCommands)
      .def("last_checkpoint_time",      &PV::PythonBindings::getLastCheckpointTime)
      .def("checkpoint",                &PV::PythonBindings::checkpoint)
      ;
}


#endif
