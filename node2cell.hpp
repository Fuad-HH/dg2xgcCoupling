#include <iostream>
#include <string>

#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_vector.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mesh.hpp>

namespace o = Omega_h;

void set_sinxcosy_tag(o::Mesh& mesh);

void node_average2cell(o::Mesh& mesh);

void render(o::Mesh& mesh, int iter, int comm_rank);