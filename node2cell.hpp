#pragma once

#include <Omega_h_defines.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>

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

o::Real calculate_l1_error(o::Mesh& mesh, std::string apporx_field_name,
                           std::string exact_field_name);

o::Real calculate_rel_l2_error(o::Mesh& mesh, std::string apporx_field_name,
                               std::string exact_field_name);

o::Real get_face_field_integral(o::Mesh& mesh, const std::string field_name);

OMEGA_H_DEVICE
o::Real area_tri(const o::Few<o::Vector<2>, 3>& tri_verts) {
    o::Few<o::Vector<2>, 2> basis22 = {tri_verts[1] - tri_verts[0],
                                       tri_verts[2] - tri_verts[0]};
    auto area = o::triangle_area_from_basis(basis22);
    return area;
}

o::Real calculate_integral_l2_error(o::Mesh& mesh,
                                    std::string approx_field_name,
                                    std::string exact_field_name);

o::Reals get_l_inf_error(o::Mesh& mesh, std::string approx_field_name,
                         std::string exact_field_name, o::LOs ids);

o::Reals read_field_from_file(std::string file_name);

void cell2node_degas2_style(o::Mesh& mesh, const std::string face_field_name,
                            const std::string new_field_name);

o::Reals sort_field(o::Reals field, std::string simNumbering_file);

void compute_cell_area_tag(o::Mesh& mesh);

void cell2node_degas2_style_area_weighted(o::Mesh& mesh,
                                          const std::string face_field_name,
                                          const std::string new_field_name);