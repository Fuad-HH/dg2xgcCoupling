#include "node2cell.hpp"

#include <pcms/pcms.h>

#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_vector.hpp>

void set_sinxcosy_tag(o::Mesh& mesh) {
    // get the bounding box of the mesh
    Omega_h::BBox<2> bb = Omega_h::get_bounding_box<2>(&mesh);
    printf("Bounding box: min=(%f %f) max=(%f %f)\n", bb.min[0], bb.min[1],
           bb.max[0], bb.max[1]);
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
