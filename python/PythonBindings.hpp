#ifndef PYTHONBINDINGS_HPP_
#define PYTHONBINDINGS_HPP_

#define PYTHON_MODULE_NAME      PythonBindings 
#define PYTHON_MODULE_NAME_STR "PythonBindings"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <bindings/Commander.hpp>
#include <bindings/PVData.hpp>
#include <bindings/PVData.hpp>

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
   void                      begin();
   double                    advance(unsigned int steps);
   void                      finish();
   py::array_t<float>        getConnectionWeights(const char *connName); 
   void                      setConnectionWeights(const char *connName, py::array_t<float> *data); 
   std::shared_ptr<DataPack> getLayerSparseActivity(const char *layerName);
   std::shared_ptr<DataPack> getLayerActivity(const char *layerName); 
   std::shared_ptr<DataPack> getLayerState(const char *layerName); 
   void                      setLayerState(const char *layerName, std::shared_ptr<DataPack> data);
   bool                      isFinished();
   py::array_t<double>       getProbeValues(const char *probeName);
   bool                      isRoot();
   void                      waitForCommands();
   double                    getLastCheckpointTime();
   void                      checkpoint();
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

   py::class_<PV::DataPack, std::shared_ptr<PV::DataPack>> datapack(m, "DataPack");
      datapack.def("format",
           &PV::DataPack::format,
           "Returns the format used to pack data");
      datapack.def("get",
           &PV::DataPack::get,
           "Returns the value at the specified location.");
      datapack.def("set",
           (bool (PV::DataPack::*)(int, int, int, int, float))&PV::DataPack::set,
           "Set a single value at the specified location");
      datapack.def("set",
           (bool (PV::DataPack::*)(int, PV::SparseVector*))&PV::DataPack::set,
           "Set the contents of the specified batch to the given vector.");
      datapack.def("set",
           (bool (PV::DataPack::*)(int, PV::DenseVector*))&PV::DataPack::set,
           "Set the contents of the specified batch to the given vector.");
      datapack.def("as_sparse",
           &PV::DataPack::asSparse,
           "Returns a copy of the packed data in sparse format.");
      datapack.def("as_dense",
           &PV::DataPack::asDense,
           "Returns a copy of the packed data in dense format.");
      datapack.def_property_readonly("npatch", &PV::DataPack::getNB); /* for connections */
      datapack.def_property_readonly("nbatch", &PV::DataPack::getNB); /* for layers */
      datapack.def_property_readonly("ny", &PV::DataPack::getNY);
      datapack.def_property_readonly("nx", &PV::DataPack::getNX);
      datapack.def_property_readonly("nf", &PV::DataPack::getNF);
      datapack.def_property_readonly("nelements", &PV::DataPack::elements);
      datapack.def("__repr__",
            [](const PV::DataPack &d) {
               return std::string("<") + d.format()
                     + std::string(" OpenPV.DataPack with dimensions (")
                     + std::to_string(d.getNB()) + std::string(", ")
                     + std::to_string(d.getNY()) + std::string(", ")
                     + std::to_string(d.getNX()) + std::string(", ")
                     + std::to_string(d.getNF()) + std::string(")>");
            });

   py::class_<PV::SparsePack, std::shared_ptr<PV::SparsePack>>(m, "SparsePack", datapack)
      .def(py::init<int, int, int, int>())
      .def("__getstate__",
               [](const PV::SparsePack &d) {
                  py::list l;
                  for (int i = 0; i < d.getNB(); i++) {
                     l.append(d.asSparse(i));
                  }
                  return py::make_tuple(l, d.getNY(), d.getNX(), d.getNF());
               })
      .def("__setstate__",
               [](py::tuple t) {
                  PV::SparsePack p((int)py::cast<int>(py::len(t[0])).cast<int>(), t[1].cast<int>(), t[2].cast<int>(), t[3].cast<int>());
                  for (int i = 0; i < p.getNB(); i++) {
                     py::list l = t[0];
                     PV::SparseVector v = l.cast<PV::SparseVector>();
                     p.set(i, &v);
                  }
                  return p;
               })
   ;
   py::class_<PV::DensePack, std::shared_ptr<PV::DensePack>>(m, "DensePack", datapack)
      .def(py::init<int, int, int, int>())
      .def(py::pickle(
               [](const PV::DensePack &d) {
                  py::list l;
                  for (int i = 0; i < d.getNB(); i++) {
                     l.append(d.asDense(i));
                  }
                  return py::make_tuple(l, d.getNY(), d.getNX(), d.getNF());
               },
               [](py::tuple t) {
                  PV::DensePack p((int)py::cast<int>(py::len(t[0])).cast<int>(), t[1].cast<int>(), t[2].cast<int>(), t[3].cast<int>());
                  for (int i = 0; i < p.getNB(); i++) {
                     py::list l = t[0];
                     PV::DenseVector v = l.cast<PV::DenseVector>();
                     p.set(i, &v);
                  }
                  return p;
               }))
   ;
}


#endif
