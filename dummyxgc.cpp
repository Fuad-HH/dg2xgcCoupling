#include <mpi.h>
#include <pcms/client.h>
#include <pcms/omega_h_field.h>
#include <pcms/pcms.h>

#include <Omega_h_defines.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <iostream>
#include <string>

#include "node2cell.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <input mesh>\n";
        return -1;
    }
    std::string input_mesh = argv[1];

    o::Library library(&argc, &argv);
    o::Mesh mesh = Omega_h::binary::read(input_mesh, library.self());
    printf("Mesh loaded in **xgc** with %d elements and %d nodes\n",
           mesh.nelems(), mesh.nverts());
    set_global_tag(mesh);
    set_sinxcosy_tag(mesh);

    // check if it has the tag
    if (mesh.has_tag(o::VERT, "sinxcosy")) {
        printf("The mesh has the tag sinxcosy\n");
    } else {
        printf("The mesh does not have the tag sinxcosy\n");
    }

    printf("Now the mesh data is being sent to the coupler...\n");
    pcms::CouplerClient cpl("xgcClient", MPI_COMM_WORLD);
    cpl.AddField("sincos",
                 pcms::OmegaHFieldAdapter<Omega_h::Real>("sinxcosy", mesh));
    cpl.BeginSendPhase();
    cpl.SendField("sincos");
    cpl.EndSendPhase();

    printf("The mesh data is sent to the coupler...\n");

    return 0;
}