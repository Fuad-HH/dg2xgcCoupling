#pragma once

#include <Omega_h_mesh.hpp>

namespace o = Omega_h;

void set_sinxcosy_tag(o::Mesh& mesh);

void node_average2cell(o::Mesh& mesh);

void node2cell(o::Mesh& mesh, std::string node_field_name,
               std::string face_field_name);

void render(o::Mesh& mesh, int iter, int comm_rank);

void set_n_sq_tag(o::Mesh& mesh);

void set_global_tag(o::Mesh& mesh);

void cell2node(o::Mesh& mesh, std::string field_name,
               std::string new_field_name, o::Real radius);

o::Real calculate_l2_error(o::Mesh& mesh, std::string apporx_field_name,
                           std::string exact_field_name);

o::Real calculate_rel_l2_error(o::Mesh& mesh, std::string apporx_field_name,
                               std::string exact_field_name);
