/************************ File Description ******************************
* Brief: This the dummy degas2 app that demonstrates how to use the PCMS library
*        to couple two applications.
* Functionality: It creates a data that lives on the mesh faces (as face tags)
                 and sends the data to the coupler. I also receives the data
sent by the coupler which it got from xgc.
**************************************************************************/
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
    printf("Mesh loaded in **degas2** app with %d elements\n", mesh.nelems());
    set_global_tag(mesh);
    o::Read<o::Real> data(mesh.nverts(), 0.0);
    // mesh.add_tag<o::Real>(0, "sinxcosy", 1, data);
    set_sinxcosy_tag(mesh);
    node_average2cell(mesh);

    pcms::CouplerClient cpl("degas2Client", MPI_COMM_WORLD);
    cpl.AddField("sincos",
                 pcms::OmegaHFieldAdapter<Omega_h::Real>("sinxcosy", mesh));
    cpl.BeginReceivePhase();
    cpl.ReceiveField("sincos");
    cpl.EndReceivePhase();
    printf("The mesh data is received to the degas2 solver...\n");

    set_n_sq_tag(mesh);
    if (!mesh.has_tag(o::FACE, "n_sq")) {
        printf("The mesh does not have the tag n_sq\n");
    }

    cpl.AddField("n_sq", pcms::OmegaHFieldAdapter<Omega_h::Real>(
                             "sinxcosy", mesh, "", 10, 10,
                             pcms::detail::mesh_entity_type::FACE));
    cpl.BeginSendPhase();
    cpl.SendField("n_sq");
    cpl.EndSendPhase();

    // now check if the tag is received
    if (mesh.has_tag(o::VERT, "sinxcosy")) {
        printf("The mesh has the tag sinxcosy\n");
    } else {
        printf("The mesh does not have the tag sinxcosy\n");
    }

    printf("The mesh data is received to the degas2 solver...\n");

    return 0;
}