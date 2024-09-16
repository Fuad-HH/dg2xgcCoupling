#pragma once

#include <Omega_h_mesh.hpp>

namespace o = Omega_h;

void set_sinxcosy_tag(o::Mesh& mesh);

void node_average2cell(o::Mesh& mesh);

void render(o::Mesh& mesh, int iter, int comm_rank);

void set_n_sq_tag(o::Mesh& mesh);

void set_global_tag(o::Mesh& mesh);
