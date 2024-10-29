#include <Omega_h_build.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_file.hpp>
#include <pcms/interpolator/MLSInterpolation.hpp>
#include <pcms/interpolator/adj_search.hpp>

namespace o = Omega_h;

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

int main(int argc, char** argv) {
    o::Library library(&argc, &argv);
    if (argc != 4) {
        printf("Usage: %s <source mesh> <target mesh> <interpolation radius>\n",
               argv[0]);
        return -1;
    }
    std::string source_mesh_fname = argv[1];
    std::string target_mesh_fname = argv[2];
    if (source_mesh_fname == "internal" || target_mesh_fname == "internal") {
        OMEGA_H_CHECK_PRINTF(target_mesh_fname == source_mesh_fname,
                             "Both meshes must be internal but got %s and %s\n",
                             target_mesh_fname.c_str(),
                             source_mesh_fname.c_str());
    }
    o::Real interpolation_radius = std::stod(argv[3]);

    printf("Source mesh: %s\n", source_mesh_fname.c_str());
    printf("Target mesh: %s\n", target_mesh_fname.c_str());
    printf("Interpolation radius: %f\n", interpolation_radius);

    o::Mesh source_mesh;
    o::Mesh target_mesh;
    if (source_mesh_fname != "internal") {
        source_mesh = o::binary::read(source_mesh_fname, library.self());
        target_mesh = o::binary::read(target_mesh_fname, library.self());
    } else {
        source_mesh = o::build_box(library.world(), OMEGA_H_SIMPLEX, 1, 1, 1,
                                   100, 100, 0, false);
        target_mesh = o::build_box(library.world(), OMEGA_H_SIMPLEX, 1, 1, 1,
                                   50, 50, 0, false);
    }

    printf("________________ Mesh Loaded ________________\n");
    printf("Source mesh has %d vertices and %d elements\n",
           source_mesh.nverts(), source_mesh.nelems());
    printf("Target mesh has %d vertices and %d elements\n",
           target_mesh.nverts(), target_mesh.nelems());
    printf("_____________________________________________\n");

    printf("________________ Set Exact Field ________________\n");
    set_sinxcosy_tag(source_mesh);
    set_sinxcosy_tag(target_mesh);
    printf("_____________________________________________\n");

    const auto& tartget_coords = target_mesh.coords();
    const auto& source_coords = source_mesh.coords();

    o::Read<o::Real> source_values =
        source_mesh.get_array<o::Real>(o::VERT, "sinxcosy");

    o::Real radius2 = interpolation_radius * interpolation_radius;
    o::Read<o::Real> radii2(target_mesh.nverts(), radius2, "radii2");

    printf("________________ MLS Interpolation ________________\n");
    SupportResults support =
        searchNeighbors(source_mesh, target_mesh, radius2, 12, false);
    auto approx_target_values = mls_interpolation(
        source_values, source_coords, tartget_coords, support, 2, 2, radius2);
    printf("_____________________________________________\n");

    printf("________________ Write Approx Field ________________\n");
    target_mesh.add_tag<o::Real>(o::VERT, "sinxcosy_approx", 1,
                                 o::Reals(approx_target_values));

    // write the target mesh as vtk
    o::vtk::write_parallel("fine2coarse_interpolation_target.vtk", &target_mesh,
                           2);
    o::vtk::write_parallel("fine2coarse_interpolation_source.vtk", &source_mesh,
                           2);
    printf("______________________ Done _______________________\n");
    return 0;
}