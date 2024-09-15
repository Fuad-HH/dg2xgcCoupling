#ifndef DG2XGCFIELDADAPTER_H
#define DG2XGCFIELDADAPTER_H

#include <pcms/array_mask.h>
#include <pcms/assert.h>
#include <pcms/field.h>
#include <pcms/memory_spaces.h>
#include <pcms/profile.h>
#include <pcms/types.h>

#include <Omega_h_mesh.hpp>
#include <string>
#include <vector>

namespace pcms {
namespace detail {
class dg2xgcFieldAdapter {
   public:
    using memory_space = HostMemorySpace;
    using value_type = Real;

    dg2xgcFieldAdapter(std::string name, Omega_h::Mesh* mesh,
                       Omega_h::Reals& data)
        : name_(std::move(name)), mesh_(mesh), data_(data) {
        PCMS_FUNCTION_TIMER;
    }

   private:
    std::string name_;
    Omega_h::Mesh* mesh_;
    Omega_h::Reals& data_;
};

}  // namespace detail
}  // namespace pcms

#endif  // DG2XGCFIELDADAPTER_H
