/**
 * @file python/numpy_cpp.cpp
 * @brief pybind11 bridge — zero-copy via buffer protocol
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <np/np.hpp>
namespace py = pybind11;
using namespace np;

template <typename T>
ndarray<T> to_ndarray(py::array a) {
    py::array_t<T, py::array::c_style | py::array::forcecast> arr(a);
    py::buffer_info info = arr.request();
    std::vector<int> shape;
    for (auto d : info.shape) shape.push_back((int)d);
    ndarray<T> out(shape);
    std::memcpy(out.data().data(), info.ptr, out.size()*sizeof(T));
    return out;
}
template <typename T>
py::array to_pyarray(const ndarray<T>& a) {
    std::vector<ssize_t> shape, strides;
    for (auto d : a.shape) shape.push_back(d);
    ssize_t s = sizeof(T);
    for (int i=(int)a.shape.size()-1;i>=0;--i){ strides.push_back(s); s*=a.shape[i];}
    std::reverse(strides.begin(), strides.end());
    auto* vec = new std::vector<T>(a.data().begin(), a.data().end());
    py::capsule cap(vec, [](void* p){ delete reinterpret_cast<std::vector<T>*>(p); });
    return py::array(py::buffer_info(vec->data(), sizeof(T), py::format_descriptor<T>::format(), a.ndim(), shape, strides), cap);
}
template<>
inline ndarray<bool> to_ndarray<bool>(py::array a) {
    py::array_t<bool, py::array::c_style | py::array::forcecast> arr(a);
    py::buffer_info info = arr.request();
    std::vector<int> shape;
    for (auto d: info.shape) shape.push_back((int)d);
    ndarray<bool> out(shape);
    bool* ptr = static_cast<bool*>(info.ptr);
    for(size_t i=0;i<out.size();++i) out.data()[i]=ptr[i];
    return out;
}
template<>
inline py::array to_pyarray<bool>(const ndarray<bool>& a){
    std::vector<ssize_t> shape, strides;
    for(auto d: a.shape) shape.push_back(d);
    ssize_t s=sizeof(bool);
    for(int i=(int)a.shape.size()-1;i>=0;--i){strides.push_back(s); s*=a.shape[i];}
    std::reverse(strides.begin(), strides.end());
    auto* buf = new std::vector<unsigned char>(a.size());
    for(size_t i=0;i<a.size();++i) (*buf)[i]= a.data()[i]?1:0;
    py::capsule cap(buf, [](void*p){ delete reinterpret_cast<std::vector<unsigned char>*>(p); });
    return py::array(py::buffer_info(buf->data(), sizeof(bool), py::format_descriptor<bool>::format(), a.ndim(), shape, strides), cap);
}

template <typename T>
void bind_ndarray(py::module& m, const char* name){
    if constexpr (std::is_same_v<T, bool>) {
        py::class_<ndarray<T>>(m, name)
            .def(py::init<std::vector<int>>())
            .def_property_readonly("shape", [](const ndarray<T>& a){ return a.shape; })
            .def_property_readonly("ndim", [](const ndarray<T>& a){ return a.ndim(); })
            .def_property_readonly("size", [](const ndarray<T>& a){ return a.size(); })
            .def("copy", [](const ndarray<T>& a){ return a.copy(); });
    } else {
        py::class_<ndarray<T>>(m, name, py::buffer_protocol())
            .def(py::init<std::vector<int>>())
            .def_property_readonly("shape", [](const ndarray<T>& a){ return a.shape; })
            .def_property_readonly("ndim", [](const ndarray<T>& a){ return a.ndim(); })
            .def_property_readonly("size", [](const ndarray<T>& a){ return a.size(); })
            .def("copy", [](const ndarray<T>& a){ return a.copy(); })
            .def_buffer([](ndarray<T>& a) -> py::buffer_info{
                std::vector<ssize_t> shape, strides;
                for(auto d: a.shape) shape.push_back(d);
                ssize_t s=sizeof(T);
                for(int i=(int)a.shape.size()-1;i>=0;--i){strides.push_back(s); s*=a.shape[i];}
                std::reverse(strides.begin(), strides.end());
                return py::buffer_info(a.data().data(), sizeof(T), py::format_descriptor<T>::format(), a.ndim(), shape, strides);
            });
    }
}

py::array sin_wrapper(py::array a){ return to_pyarray(sin(to_ndarray<double>(a))); }
py::array cos_wrapper(py::array a){ return to_pyarray(cos(to_ndarray<double>(a))); }
py::array exp_wrapper(py::array a){ return to_pyarray(exp(to_ndarray<double>(a))); }
py::array log_wrapper(py::array a){ return to_pyarray(log(to_ndarray<double>(a))); }
py::array sqrt_wrapper(py::array a){ return to_pyarray(sqrt(to_ndarray<double>(a))); }
py::array matmul_wrapper(py::array a, py::array b){ return to_pyarray(linalg::matmul(to_ndarray<double>(a), to_ndarray<double>(b))); }
py::array dot_wrapper(py::array a, py::array b){ return to_pyarray(linalg::dot(to_ndarray<double>(a), to_ndarray<double>(b))); }
py::array inv_wrapper(py::array a){ return to_pyarray(linalg::inv(to_ndarray<double>(a))); }
py::array fft_wrapper(py::array a){ return to_pyarray(fft::fft(to_ndarray<std::complex<double>>(a))); }
py::array ifft_wrapper(py::array a){ return to_pyarray(fft::ifft(to_ndarray<std::complex<double>>(a))); }
py::array rfft_wrapper(py::array a){ return to_pyarray(fft::rfft(to_ndarray<double>(a))); }
py::array sort_wrapper(py::array a, int axis){ return to_pyarray(sort(to_ndarray<double>(a), axis)); }
py::array argsort_wrapper(py::array a, int axis){ return to_pyarray(argsort(to_ndarray<double>(a), axis)); }
py::array hbm_wrapper(py::array a){ return to_pyarray(mem::migrate_to_hbm(to_ndarray<float>(a)).data); }
py::array encode_wrapper(py::array a){ auto nd = to_ndarray<float>(a); auto ev = spike::encode_rate(nd); (void)ev; return to_pyarray(nd); }
py::array fp8_wrapper(py::array a, py::array b){ return to_pyarray(tensor::matmul_fp8(to_ndarray<float>(a), to_ndarray<float>(b))); }
py::array plus_state_wrapper(int n){ auto s = quantum::QuantumFactory::plus_state(n); std::vector<int> shape{(int)s.amps.size()}; ndarray<std::complex<double>> out(shape); for(size_t i=0;i<s.amps.size();++i) out.data()[i]=s.amps[i]; return to_pyarray(out); }
double linalg_norm_wrapper(py::array a){ return linalg::norm(to_ndarray<double>(a)); }
double linalg_det_wrapper(py::array a){ return linalg::det(to_ndarray<double>(a)); }
py::tuple linalg_svd_wrapper(py::array a){ auto nd=to_ndarray<double>(a); auto r=linalg::svd(nd); return py::make_tuple(to_pyarray(r.u), to_pyarray(r.s), to_pyarray(r.vh)); }
py::tuple linalg_eig_wrapper(py::array a){ auto nd=to_ndarray<double>(a); auto r=linalg::eig(nd); return py::make_tuple(to_pyarray(r.w), to_pyarray(r.v)); }

PYBIND11_MODULE(numpy_cpp, m){
  m.doc() = "numpy-cpp Python bridge — header-only C++20 NumPy 2.2 via pybind11 (zero-copy buffer)";
  py::enum_<dtype>(m, "dtype")
    .value("int8", dtype::int8).value("int16", dtype::int16).value("int32", dtype::int32).value("int64", dtype::int64)
    .value("uint8", dtype::uint8).value("uint16", dtype::uint16).value("uint32", dtype::uint32).value("uint64", dtype::uint64)
    .value("float32", dtype::float32).value("float64", dtype::float64).value("complex64", dtype::complex64).value("complex128", dtype::complex128)
    .value("bool_", dtype::bool_).value("bigint", dtype::bigint).export_values();
  bind_ndarray<float>(m, "ndarray_float32");
  bind_ndarray<double>(m, "ndarray_float64");
  bind_ndarray<int>(m, "ndarray_int32");
  bind_ndarray<int64_t>(m, "ndarray_int64");
  bind_ndarray<std::complex<double>>(m, "ndarray_complex128");
  bind_ndarray<bool>(m, "ndarray_bool");
  m.def("arange", [](double s,double e,double step){ return to_pyarray(arange<double>(s,e,step)); }, py::arg("start"), py::arg("stop"), py::arg("step")=1.0);
  m.def("linspace", [](double s,double e,int n){ return to_pyarray(linspace<double>(s,e,n)); });
  m.def("zeros", [](std::vector<int> shape, std::string dt){ if(dt=="float32") return to_pyarray(zeros<float>(shape)); return to_pyarray(zeros<double>(shape)); }, py::arg("shape"), py::arg("dtype")="float64");
  m.def("ones", [](std::vector<int> shape){ return to_pyarray(ones<double>(shape)); });
  m.def("eye", [](int n){ return to_pyarray(eye<double>(n)); });
  m.def("full", [](std::vector<int> shape,double v){ return to_pyarray(full<double>(shape,v)); });
  m.def("sin", &sin_wrapper);
  m.def("cos", &cos_wrapper);
  m.def("exp", &exp_wrapper);
  m.def("log", &log_wrapper);
  m.def("sqrt", &sqrt_wrapper);
  auto mlinalg = m.def_submodule("linalg", "np::linalg");
  mlinalg.def("matmul", &matmul_wrapper);
  mlinalg.def("dot", &dot_wrapper);
  mlinalg.def("norm", &linalg_norm_wrapper);
  mlinalg.def("det", &linalg_det_wrapper);
  mlinalg.def("inv", &inv_wrapper);
  mlinalg.def("svd", &linalg_svd_wrapper);
  mlinalg.def("eig", &linalg_eig_wrapper);
  auto mfft = m.def_submodule("fft", "np::fft");
  mfft.def("fft", &fft_wrapper);
  mfft.def("ifft", &ifft_wrapper);
  mfft.def("rfft", &rfft_wrapper);
  auto mrandom = m.def_submodule("random", "np::random");
  mrandom.def("rand", [](std::vector<int> shape){ return to_pyarray(random::rand<double>(shape)); });
  mrandom.def("randn", [](std::vector<int> shape){ return to_pyarray(random::randn<double>(shape)); });
  mrandom.def("randint", [](int l,int h,std::vector<int> shape){ return to_pyarray(random::randint<int>(l,h,shape)); });
  auto msort = m.def_submodule("sorting", "np::sorting");
  msort.def("sort", &sort_wrapper, py::arg("a"), py::arg("axis")=-1);
  msort.def("argsort", &argsort_wrapper, py::arg("a"), py::arg("axis")=-1);
  auto mlattice = m.def_submodule("lattice", "np::lattice");
  mlattice.def("cubic", [](int n){ return lattice::LatticeFactory::cubic<double>(n); });
  mlattice.def("lll", [](const lattice::Lattice<double>& lat){ return lat.lll_reduce(); });
  auto mpadic = m.def_submodule("padic", "np::padic");
  mpadic.def("padic", [](int p,int64_t v,int prec){ return padic::Padic<int64_t>(p,v,prec); });
  mpadic.def("valuation", [](const padic::Padic<int64_t>& a){ return a.valuation(); });
  auto mhw = m.def_submodule("hardware", "accelerator/neuromorphic/tensor/mem");
  mhw.def("hbm_migrate", &hbm_wrapper);
  auto mneuro = mhw.def_submodule("neuromorphic", "Loihi/SpiNNaker");
  mneuro.def("encode_rate", &encode_wrapper);
  auto mtensor = mhw.def_submodule("tensor", "Hopper/AMX");
  mtensor.def("matmul_fp8", &fp8_wrapper);
  auto mquantum = mhw.def_submodule("quantum", "StateVector");
  mquantum.def("plus_state", &plus_state_wrapper);
  m.def("dtype_of", [](py::array a){ return a.dtype().kind(); });
  m.def("promote_types", [](dtype a,dtype b){ return promote_types(a,b); });
}
