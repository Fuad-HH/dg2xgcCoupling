#include "node2cell.hpp"

#include <Omega_h_macros.h>
#include <pcms/pcms.h>

#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_reduce.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_vector.hpp>
#include <cstdio>
#include <fstream>
#include <pcms/interpolator/MLSInterpolation.hpp>
#include <pcms/interpolator/adj_search.hpp>
#include <string>

o::Real calculate_l2_error(o::Mesh& mesh, std::string apporx_field_name,
                           std::string exact_field_name) {
    OMEGA_H_CHECK(mesh.has_tag(o::VERT, apporx_field_name));
    OMEGA_H_CHECK(mesh.has_tag(o::VERT, exact_field_name));
    auto approx_field = mesh.get_array<o::Real>(o::VERT, apporx_field_name);
    auto exact_field = mesh.get_array<o::Real>(o::VERT, exact_field_name);

    auto class_id = mesh.get_array<o::LO>(o::VERT, "class_id");

    o::Real l2_error = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, exact_field.size()),
        KOKKOS_LAMBDA(const LO node_id, o::Real& l2) {
            if (class_id[node_id] < 81) {
                o::Real diff = approx_field[node_id] - exact_field[node_id];
                l2 += diff * diff;
            }
        },
        Kokkos::Sum<o::Real>(l2_error));

    // get the size of the exact field
    o::LO size = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, exact_field.size()),
        KOKKOS_LAMBDA(const LO node_id, o::LO& l2) {
            if (class_id[node_id] < 81) {
                l2++;
            }
        },
        Kokkos::Sum<o::LO>(size));

    o::Real exact_sum2 = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, exact_field.size()),
        KOKKOS_LAMBDA(const LO node_id, o::Real& l2) {
            if (class_id[node_id] < 81) {
                l2 += exact_field[node_id] * exact_field[node_id];
            }
        },
        Kokkos::Sum<o::Real>(exact_sum2));

    l2_error = Kokkos::sqrt(l2_error) / Kokkos::sqrt(exact_sum2 * size);

    return l2_error;
}

o::Real calculate_l1_error(o::Mesh& mesh, std::string apporx_field_name,
                           std::string exact_field_name) {
    OMEGA_H_CHECK(mesh.has_tag(o::VERT, apporx_field_name));
    OMEGA_H_CHECK(mesh.has_tag(o::VERT, exact_field_name));
    auto approx_field = mesh.get_array<o::Real>(o::VERT, apporx_field_name);
    auto exact_field = mesh.get_array<o::Real>(o::VERT, exact_field_name);

    o::Real l1_error = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, exact_field.size()),
        KOKKOS_LAMBDA(const LO node_id, o::Real& l2) {
            o::Real diff = approx_field[node_id] - exact_field[node_id];
            l2 += Kokkos::abs(diff);
        },
        Kokkos::Sum<o::Real>(l1_error));

    l1_error = l1_error / exact_field.size();

    return l1_error;
}

o::Real calculate_rel_l2_error(o::Mesh& mesh, std::string apporx_field_name,
                               std::string exact_field_name) {
    OMEGA_H_CHECK(mesh.has_tag(o::VERT, apporx_field_name));
    OMEGA_H_CHECK(mesh.has_tag(o::VERT, exact_field_name));
    auto approx_field = mesh.get_array<o::Real>(o::VERT, apporx_field_name);
    auto exact_field = mesh.get_array<o::Real>(o::VERT, exact_field_name);

    o::Real l2_error = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, exact_field.size()),
        KOKKOS_LAMBDA(const LO node_id, o::Real& l2) {
            o::Real diff = approx_field[node_id] - exact_field[node_id];
            diff = diff / exact_field[node_id];
            l2 += diff * diff;
        },
        Kokkos::Sum<o::Real>(l2_error));

    l2_error = Kokkos::sqrt(l2_error) / exact_field.size();

    return l2_error;
}

void cell2node(o::Mesh& mesh, const std::string field_name,
               const std::string new_field_name, o::Real radius) {
    o::Real radius_sq = radius * radius;
    auto coords = mesh.coords();
    o::Write<o::Real> source_locations(mesh.nfaces() * 2, 0.0);

    auto face2node = mesh.ask_down(o::FACE, o::VERT).ab2b;
    o::parallel_for(
        mesh.nfaces(), OMEGA_H_LAMBDA(o::LO face) {
            auto nodes = o::gather_verts<3>(face2node, face);
            o::Few<o::Vector<2>, 3> face_coords =
                o::gather_vectors<3, 2>(coords, nodes);
            o::Vector<2> centroid = o::average(face_coords);
            source_locations[2 * face + 0] = centroid[0];
            source_locations[2 * face + 1] = centroid[1];
        });

    SupportResults support = searchNeighbors(mesh, radius_sq, 15, true);
    o::Reals field = mesh.get_array<o::Real>(2, field_name);
    auto interpolated_values = mls_interpolation(
        field, source_locations, coords, support, 2, 2, support.radii2);
    printf("Interpolated values size: %d\n", interpolated_values.size());
    printf("Number of vertex nodes: %d\n", mesh.nverts());
    mesh.add_tag<o::Real>(o::VERT, new_field_name, 1,
                          o::Reals(interpolated_values));

    o::Write<o::LO> num_supports(mesh.nverts(), -1);
    o::parallel_for(
        mesh.nverts(), OMEGA_H_LAMBDA(o::LO node) {
            num_supports[node] =
                support.supports_ptr[node + 1] - support.supports_ptr[node];
        });
    mesh.add_tag<o::LO>(o::VERT, "num_supports", 1, o::LOs(num_supports));
}

void cell2node_degas2_style(o::Mesh& mesh, const std::string face_field_name,
                            const std::string new_field_name) {
    auto node2faceFace = mesh.ask_up(o::VERT, o::FACE).ab2b;
    auto node2faceOffset = mesh.ask_up(o::VERT, o::FACE).a2ab;

    OMEGA_H_CHECK(mesh.has_tag(o::FACE, face_field_name));
    auto original_field = mesh.get_array<o::Real>(o::FACE, face_field_name);

    o::Write<o::Real> interpolated_field(mesh.nverts(), 0.0);

    o::parallel_for(
        mesh.nverts(), OMEGA_H_LAMBDA(o::LO node) {
            o::LO faceBegin = node2faceOffset[node];
            int nfaces = node2faceOffset[node + 1] - faceBegin;
            o::Real sum = 0.0;
            for (int i = 0; i < nfaces; i++) {
                sum += original_field[node2faceFace[faceBegin + i]];
            }
            interpolated_field[node] = sum / nfaces;
        });

    mesh.add_tag(o::VERT, new_field_name, 1, o::Reals(interpolated_field));
}

void cell2node_degas2_style_area_weighted(o::Mesh& mesh,
                                          const std::string face_field_name,
                                          const std::string new_field_name) {
    if (!mesh.has_tag(o::FACE, "cell_area")) {
        printf("'cell_area' tag not found! Computing cell area tag\n");
        compute_cell_area_tag(mesh);
    }
    o::Reals cell_area = mesh.get_array<o::Real>(o::FACE, "cell_area");
    auto node2faceFace = mesh.ask_up(o::VERT, o::FACE).ab2b;
    auto node2faceOffset = mesh.ask_up(o::VERT, o::FACE).a2ab;

    OMEGA_H_CHECK(mesh.has_tag(o::FACE, face_field_name));
    auto original_field = mesh.get_array<o::Real>(o::FACE, face_field_name);

    o::Write<o::Real> interpolated_field(mesh.nverts(), 0.0);

    o::parallel_for(
        mesh.nverts(), OMEGA_H_LAMBDA(o::LO node) {
            auto faceBegin = node2faceOffset[node];
            auto faceEnd = node2faceOffset[node + 1];
            o::Real sum = 0.0;
            o::Real area_sum = 0.0;
            for (auto face = faceBegin; face < faceEnd; ++face) {
                sum += original_field[node2faceFace[face]] *
                       cell_area[node2faceFace[face]];
                area_sum += cell_area[node2faceFace[face]];
            }
            interpolated_field[node] = sum / area_sum;
        });
    mesh.add_tag(o::VERT, new_field_name, 1, o::Reals(interpolated_field));
}

void set_sinxcosy_tag(o::Mesh& mesh) {
    // get the bounding box of the mesh
    Omega_h::BBox<2> bb = Omega_h::get_bounding_box<2>(&mesh);
    double dx = bb.max[0] - bb.min[0];
    double dy = bb.max[1] - bb.min[1];

    // loop over each node and set the tag as sin(x)cos(y)
    auto coords = mesh.coords();
    auto nnodes = mesh.nverts();
    o::Write<o::Real> sinxcosytag(nnodes);

    auto assignSinCos = OMEGA_H_LAMBDA(int node) {
        auto x = (coords[2 * node + 0] - bb.min[0]) / dx * 2.0 * M_PI;
        auto y = (coords[2 * node + 1] - bb.min[1]) / dy * 2.0 * M_PI;
        sinxcosytag[node] = 2 + sin(x) * cos(y);
    };
    o::parallel_for(nnodes, assignSinCos, "assignSinCos");

    // add the tag to the mesh
    mesh.add_tag(o::VERT, "sinxcosy", 1, o::Reals(sinxcosytag));
}

void set_n_sq_tag(o::Mesh& mesh) {
    o::BBox<2> bb = o::get_bounding_box<2>(&mesh);
    double dx = bb.max[0] - bb.min[0];
    double dy = bb.max[1] - bb.min[1];

    auto nfaces = mesh.nfaces();
    o::Write<o::Real> nSqTag(nfaces);

    auto assign_n_sq = OMEGA_H_LAMBDA(o::LO face) {
        nSqTag[face] = o::Real(face) * o::Real(face);
    };
    o::parallel_for(nfaces, assign_n_sq, "assign_n_sq");

    mesh.add_tag(o::FACE, "n_sq", 1, o::Reals(nSqTag));
}

void set_global_tag(o::Mesh& mesh) {
    if (!mesh.has_tag(o::VERT, "global")) {
        o::Write<o::GO> globalTag(mesh.nverts());
        o::parallel_for(
            mesh.nverts(),
            OMEGA_H_LAMBDA(o::GO node) { globalTag[node] = node; });
        mesh.add_tag(o::VERT, "global", 1, o::GOs(globalTag));
    }

    if (!mesh.has_tag(o::FACE, "global")) {
        o::Write<o::GO> globalTag(mesh.nfaces());
        o::parallel_for(
            mesh.nfaces(),
            OMEGA_H_LAMBDA(o::GO face) { globalTag[face] = face; });
        mesh.add_tag(o::FACE, "global", 1, o::GOs(globalTag));
    }
}

void node_average2cell(o::Mesh& mesh) {
    auto sinxcosytag = mesh.get_array<o::Real>(o::VERT, "sinxcosy");
    auto face2node = mesh.ask_down(o::FACE, o::VERT).ab2b;
    auto coords = mesh.coords();
    o::LO nfaces = mesh.nfaces();
    o::Write<o::Real> sinxcosyCell(nfaces);

    auto averageSinCos = OMEGA_H_LAMBDA(o::LO face) {
        auto faceNodes = o::gather_verts<3>(face2node, face);
        o::Real sum = sinxcosytag[faceNodes[0]] + sinxcosytag[faceNodes[1]] +
                      sinxcosytag[faceNodes[2]];
        sinxcosyCell[face] = sum / 3.0;
    };
    o::parallel_for(nfaces, averageSinCos, "averageSinCos");

    mesh.add_tag(o::FACE, "sinxcosy", 1, o::Reals(sinxcosyCell));
}

void node2cell(o::Mesh& mesh, std::string node_field_name,
               std::string face_field_name) {
    auto node_field = mesh.get_array<o::Real>(o::VERT, node_field_name);
    auto face2node = mesh.ask_down(o::FACE, o::VERT).ab2b;
    auto coords = mesh.coords();
    o::LO nfaces = mesh.nfaces();
    o::Write<o::Real> faceField(nfaces);

    auto averageField = OMEGA_H_LAMBDA(o::LO face) {
        auto faceNodes = o::gather_verts<3>(face2node, face);
        o::Real sum = node_field[faceNodes[0]] + node_field[faceNodes[1]] +
                      node_field[faceNodes[2]];
        faceField[face] = sum / 3.0;
    };
    o::parallel_for(nfaces, averageField, "averageField");

    mesh.add_tag(o::FACE, face_field_name, 1, o::Reals(faceField));
}

void render(o::Mesh& mesh, int iter, int comm_rank) {
    std::stringstream ss;
    ss << "coupled_mesh" << iter << "_r" << comm_rank << ".vtk";
    std::string s = ss.str();
    o::vtk::write_parallel(s, &mesh, mesh.dim());
}

void compute_cell_area_tag(o::Mesh& mesh) {
    auto coords = mesh.coords();
    auto faces2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
    auto n_faces = mesh.nfaces();
    o::Write<o::Real> face_areas(n_faces);

    o::parallel_for(
        n_faces, OMEGA_H_LAMBDA(const int i) {
            auto face_nodes = o::gather_verts<3>(faces2nodes, i);
            o::Few<o::Vector<2>, 3> face_coords;
            face_coords = o::gather_vectors<3, 2>(coords, face_nodes);
            o::Real face_area = area_tri(face_coords);
            face_areas[i] = face_area;
        });

    mesh.add_tag<o::Real>(o::FACE, "cell_area", 1, o::Reals(face_areas));
}

o::Real get_face_field_integral(o::Mesh& mesh, const std::string field_name) {
    OMEGA_H_CHECK(mesh.has_tag(o::FACE, field_name));
    const auto coords = mesh.coords();
    const auto faces2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
    const auto n_faces = mesh.nfaces();
    o::Real total_area = 0.0;
    o::Write<o::Real> face_areas(n_faces);

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, n_faces),
        KOKKOS_LAMBDA(const int i, o::Real& local_area) {
            auto face_nodes = o::gather_verts<3>(faces2nodes, i);
            o::Few<o::Vector<2>, 3> face_coords;
            face_coords = o::gather_vectors<3, 2>(coords, face_nodes);
            o::Real face_area = area_tri(face_coords);
            face_areas[i] = face_area;
            local_area += face_area;
        },
        Kokkos::Sum<o::Real>(total_area));

    auto face_field = mesh.get_array<o::Real>(o::FACE, field_name);
    o::Real field_integral = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, n_faces),
        KOKKOS_LAMBDA(const int i, o::Real& local_integral) {
            local_integral += face_field[i] * (face_areas[i]);
        },
        Kokkos::Sum<o::Real>(field_integral));

    return field_integral;
}

o::Real calculate_integral_l2_error(o::Mesh& mesh,
                                    std::string approx_field_name,
                                    std::string exact_field_name) {
    OMEGA_H_CHECK(mesh.has_tag(o::FACE, approx_field_name));
    OMEGA_H_CHECK(mesh.has_tag(o::FACE, exact_field_name));

    auto approx_field = mesh.get_array<o::Real>(o::FACE, approx_field_name);
    auto exact_field = mesh.get_array<o::Real>(o::FACE, exact_field_name);
    auto class_id = mesh.get_array<o::LO>(o::FACE, "class_id");

    const auto coords = mesh.coords();
    const auto faces2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
    const auto n_faces = mesh.nfaces();
    if (!mesh.has_tag(o::FACE, "cell_area")) {
        printf("'cell_area' tag not found! Computing cell area tag\n");
        compute_cell_area_tag(mesh);
    }
    auto face_areas = mesh.get_array<o::Real>(o::FACE, "cell_area");

    o::Write<o::Real> integral_error(n_faces);
    o::Write<o::Real> exact_face_int_2(n_faces);
    o::parallel_for(
        n_faces, OMEGA_H_LAMBDA(const int i) {
            if (class_id[i] > 100 || class_id[i] < 97) {
                o::Real diff = approx_field[i] - exact_field[i];
                o::Real error2 = diff * diff * face_areas[i] * face_areas[i];
                integral_error[i] = error2;
                exact_face_int_2[i] = exact_field[i] * exact_field[i] *
                                      face_areas[i] * face_areas[i];
            }
        });

    o::Real error_sum = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, n_faces),
        KOKKOS_LAMBDA(const int i, o::Real& local_error) {
            if (class_id[i] > 100 || class_id[i] < 97) {
                local_error += integral_error[i];
            }
        },
        Kokkos::Sum<o::Real>(error_sum));

    o::Real exact_sum = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, n_faces),
        KOKKOS_LAMBDA(const int i, o::Real& local_exacts) {
            if (class_id[i] > 100 || class_id[i] < 97) {
                local_exacts += exact_face_int_2[i];
            }
        },
        Kokkos::Sum<o::Real>(exact_sum));

    o::LO size = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, n_faces),
        KOKKOS_LAMBDA(const int i, o::LO& l2) {
            if (class_id[i] > 100 || class_id[i] < 97) {
                l2++;
            }
        },
        Kokkos::Sum<o::LO>(size));

    o::Real l2_error = Kokkos::sqrt(error_sum / (exact_sum * size));

    return l2_error;
}

o::Reals get_l_inf_error(o::Mesh& mesh, std::string approx_field_name,
                         std::string exact_field_name, o::LOs ids) {
    OMEGA_H_CHECK(mesh.has_tag(o::VERT, approx_field_name));
    OMEGA_H_CHECK(mesh.has_tag(o::VERT, exact_field_name));

    auto approx_field = mesh.get_array<o::Real>(o::VERT, approx_field_name);
    auto exact_field = mesh.get_array<o::Real>(o::VERT, exact_field_name);

    o::Write<o::Real> l_inf_error(ids.size(), 0.0);

    o::parallel_for(
        ids.size(), OMEGA_H_LAMBDA(const int i) {
            OMEGA_H_CHECK(ids[i] < approx_field.size() && ids[i] >= 0);
            l_inf_error[i] =
                std::abs(approx_field[ids[i]] - exact_field[ids[i]]);
        });

    return l_inf_error;
}

o::Reals read_field_from_file(std::string file_name) {
    // the first line is the name of the field
    // do a first pass to get the number of lines
    std::ifstream file(file_name);
    std::string line;
    int num_lines = 0;
    while (std::getline(file, line)) {
        num_lines++;
    }
    file.close();
    printf("Number of lines in file: %d\n", num_lines);

    // read the field values

    o::HostWrite<o::Real> field(num_lines - 1);
    file.open(file_name);
    std::getline(file, line);  // skip the first line
    printf("Reading field from file: %s\n", file_name.c_str());
    for (int i = 0; i < num_lines - 1; i++) {
        std::getline(file, line);
        // remove the trailing comma
        // line.pop_back();
        field[i] = std::stod(line);
    }
    file.close();
    printf("Field read from file\n");

    return o::Reals(field);
}

o::Reals sort_field(o::Reals field, std::string simNumbering_file) {
    auto simNumbering = read_field_from_file(simNumbering_file);
    OMEGA_H_CHECK(simNumbering.size() == field.size());
    o::Write<o::Real> sorted_field(field.size(), "sorted_field");

    o::parallel_for(
        field.size(), OMEGA_H_LAMBDA(const int i) {
            sorted_field[simNumbering[i]] = field[i];
        });

    return o::Reals(sorted_field);
}