#include "node2cell.hpp"

#include <Omega_h_macros.h>
#include <pcms/pcms.h>

#include <MLSInterpolation.hpp>
#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_vector.hpp>
#include <string>
// #include <adj_search_deg.hpp>
#include <adj_search_dega2.hpp>
#include <points.hpp>

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

    SupportResults support = searchNeighbors(mesh, radius_sq);
    o::Reals field = mesh.get_array<o::Real>(2, field_name);
    auto interpolated_values = mls_interpolation(field, source_locations,
                                                 coords, support, 2, radius_sq);
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
        sinxcosytag[node] = sin(x) * cos(y);
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

void render(o::Mesh& mesh, int iter, int comm_rank) {
    std::stringstream ss;
    ss << "coupled_mesh" << iter << "_r" << comm_rank << ".vtk";
    std::string s = ss.str();
    o::vtk::write_parallel(s, &mesh, mesh.dim());
}
