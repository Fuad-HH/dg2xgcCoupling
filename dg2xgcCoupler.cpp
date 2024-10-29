#include <pcms/field_evaluation_methods.h>
#include <pcms/omega_h_field.h>
#include <pcms/pcms.h>
#include <pcms/server.h>
#include <pcms/transfer_field.h>
#include <pcms/types.h>
#include <redev_partition.h>

#include <Omega_h_array.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <chrono>
#include <iostream>
#include <ratio>
#include <string>
#include <vector>

#include "node2cell.hpp"

void omega_h_coupler(MPI_Comm comm, o::Mesh& mesh,
                     o::Real interpolation_radius) {
    printf("Running omega_h_coupler... \n");
    std::vector<pcms::LO> ranks{0};
    std::vector<pcms::Real> cuts = {0};
    auto partition = redev::Partition{redev::RCBPtn{2, ranks, cuts}};
    auto& part = std::get<redev::RCBPtn>(partition);
    pcms::CouplerServer cpl("omega_h_coupler", comm, part, mesh);
    printf("CouplerServer created... \n");

    auto* dummydegas2app = cpl.AddApplication("degas2Client");
    auto* dummyXGCapp = cpl.AddApplication("xgcClient");

    o::Write<o::I8> is_overlap(mesh.nents(0), 1);
    o::Write<o::I8> is_overlap_face(mesh.nfaces(), 1);
    auto xgc_field_adapter = pcms::OmegaHFieldAdapter<pcms::Real>(
        "node_values_from_xgc", mesh, is_overlap);
    auto* field_nodedegas2 = dummydegas2app->AddField(
        "sincos", xgc_field_adapter, pcms::FieldTransferMethod::Copy,
        pcms::FieldEvaluationMethod::None, pcms::FieldTransferMethod::Copy,
        pcms::FieldEvaluationMethod::None, is_overlap);
    auto degas2_field_adapter = pcms::OmegaHFieldAdapter<pcms::Real>(
        "node_values_from_degas2", mesh, is_overlap);
    auto* field_nodeXGC = dummyXGCapp->AddField(
        "sincos", degas2_field_adapter, pcms::FieldTransferMethod::Copy,
        pcms::FieldEvaluationMethod::None, pcms::FieldTransferMethod::Copy,
        pcms::FieldEvaluationMethod::None, is_overlap);

    dummyXGCapp->ReceivePhase([&]() { field_nodeXGC->Receive(); });
    pcms::copy_field(xgc_field_adapter.GetField(),
                     degas2_field_adapter.GetField());
    dummydegas2app->SendPhase(
        [&]() { field_nodedegas2->Send(pcms::Mode::Deferred); });

    // now receive the face data from degas2
    printf("Waiting for FACE data from degas2: Mask size = %d\n",
           is_overlap_face.size());
    auto degas2_face_field_adapter = pcms::OmegaHFieldAdapter<pcms::Real>(
        "n_sq_from_degas2", mesh, is_overlap_face, "", 10, 10,
        pcms::mesh_entity_type::FACE);
    printf("FACE Field adapter created\n");
    auto* field_degas2_face = dummydegas2app->AddField(
        "n_sq", degas2_face_field_adapter, pcms::FieldTransferMethod::Copy,
        pcms::FieldEvaluationMethod::None, pcms::FieldTransferMethod::Copy,
        pcms::FieldEvaluationMethod::None, is_overlap_face);
    printf("FACE field added to coupler degas2 app\n");

    dummydegas2app->ReceivePhase([&]() { field_degas2_face->Receive(); });
    printf("FACE Data received from degas2\n");

    bool using_degas2_data = false;
    if (!using_degas2_data) {
        cell2node(mesh, "n_sq_from_degas2", "node_sinxcosy_derived",
                  interpolation_radius);

        set_sinxcosy_tag(mesh);
    }

    {
        // loop over several times to test the l2 norm change and performance
        auto start = std::chrono::steady_clock::now();
        o::Real total_time = 0.0;
        o::Real total_node2cell_time = 0.0;
        o::Real total_cell2node_time = 0.0;

        std::string exact_face_field_name = "n_sq_from_degas2";
        std::string exact_node_field_name = "sinxcosy";

        if (using_degas2_data) {
            // read the data from file
            o::Reals density_xgc = read_field_from_file(
                "/lore/hasanm4/wsources/meshes/xgcm_one_plane_output/plane_1/"
                "ghost_planes/idensity.csv");
            OMEGA_H_CHECK(density_xgc.size() == mesh.nverts());
            auto sorted_density_xgc = sort_field(
                density_xgc,
                "/lore/hasanm4/wsources/meshes/xgcm_one_plane_output/plane_1/"
                "ghost_planes/simNumbering.csv");
            mesh.add_tag<o::Real>(0, "node_sinxcosy_derived", 1,
                                  sorted_density_xgc);
            node2cell(mesh, "node_sinxcosy_derived", "exact_face_field");
            exact_face_field_name = "exact_face_field";
            exact_node_field_name = "node_sinxcosy_derived";
        }

        int num_iter = 10;
        o::Real l2_norms[num_iter];
        o::Real l1_norms[num_iter];
        o::Real int_l2_norms[num_iter];
        o::Real rel_l2_norms[num_iter];
        o::Real face_field_integral[num_iter];
        // o::LOs l_inf_locs {273773, 275797, 769745, 271800, 273859};
        o::LOs l_inf_locs{0, 1, 2, 3, 4};
        o::Real l_inf_values[10][5];
        for (int iter = 1; iter <= num_iter; iter++) {
            std::string node_field_name;
            if (iter == 1) {
                node_field_name = "node_sinxcosy_derived";
            } else {
                node_field_name =
                    "node_sinxcosy_derived_" + std::to_string(iter);
            }
            std::string face_field_name =
                "face_n_sq_derived_" + std::to_string(iter);
            std::string next_node_field_name =
                "node_sinxcosy_derived_" + std::to_string(iter + 1);

            auto start_node2cell = std::chrono::steady_clock::now();
            node2cell(mesh, node_field_name, face_field_name);
            auto end_node2cell = std::chrono::steady_clock::now();
            total_node2cell_time += std::chrono::duration<o::Real, std::milli>(
                                        end_node2cell - start_node2cell)
                                        .count();

            face_field_integral[iter - 1] =
                get_face_field_integral(mesh, face_field_name);

            auto start_cell2node = std::chrono::steady_clock::now();
            cell2node(mesh, face_field_name, next_node_field_name,
                      interpolation_radius);
            // cell2node_degas2_style(mesh, face_field_name,
            // next_node_field_name);
            auto end_cell2node = std::chrono::steady_clock::now();
            total_cell2node_time += std::chrono::duration<o::Real, std::milli>(
                                        end_cell2node - start_cell2node)
                                        .count();

            l2_norms[iter - 1] = calculate_l2_error(mesh, next_node_field_name,
                                                    exact_node_field_name);
            l1_norms[iter - 1] = calculate_l1_error(mesh, next_node_field_name,
                                                    exact_node_field_name);
            int_l2_norms[iter - 1] = calculate_integral_l2_error(
                mesh, face_field_name, exact_face_field_name);
            rel_l2_norms[iter - 1] = calculate_rel_l2_error(
                mesh, next_node_field_name, exact_node_field_name);

            auto l_inf_error = get_l_inf_error(
                mesh, next_node_field_name, exact_node_field_name, l_inf_locs);
            o::HostRead<o::Real> l_inf_error_host(l_inf_error);
            // save the l_inf error values in l_int_values
            for (int i = 0; i < l_inf_locs.size(); i++) {
                l_inf_values[iter - 1][i] = l_inf_error_host[i];
            }

            printf(
                "-------------------------- Iteration %d "
                "--------------------------\n",
                iter);
        }
        auto end = std::chrono::steady_clock::now();
        total_time =
            std::chrono::duration<o::Real, std::milli>(end - start).count();

        printf("----------Timing Info----------\n");
        printf("Total time: %f ms\n", total_time);
        printf("Total node2cell time: %f ms\n", total_node2cell_time);
        printf("Average node2cell time: %f ms\n",
               total_node2cell_time / num_iter);
        printf("Total cell2node time: %f ms\n", total_cell2node_time);
        printf("Average cell2node time: %f ms\n",
               total_cell2node_time / num_iter);

        printf("\n----------L2 Norms----------\n");
        printf("Abs L2 norms: \n");
        for (int i = 0; i < num_iter; i++) {
            printf(" ,%.10f", l2_norms[i]);
        }
        printf("\n-----------------l1 Norms-----------------\n");
        for (int i = 0; i < num_iter; i++) {
            printf(" ,%.10f", l1_norms[i]);
        }
        printf("\nRel L2 norms: \n");
        for (int i = 0; i < num_iter; i++) {
            printf(" ,%.10f", rel_l2_norms[i]);
        }
        printf("\n");

        printf("\n----------Face Field Integrals----------\n");
        for (int i = 0; i < num_iter; i++) {
            printf(" ,%.10f", face_field_integral[i]);
        }

        printf("\n----------Integral L2 Norms----------\n");
        for (int i = 0; i < num_iter; i++) {
            printf(" ,%.10f", int_l2_norms[i]);
        }

        printf("\n----------L_inf Norms----------\n");
        // as a list of lists
        for (int i = 0; i < num_iter; i++) {
            for (int j = 0; j < l_inf_locs.size(); j++) {
                printf(" ,%.10f", l_inf_values[i][j]);
            }
            printf("\n");
        }
    }

    Omega_h::vtk::write_parallel("degas2_coupling_result.vtk", &mesh,
                                 mesh.dim());
}

int main(int argc, char** argv) {
    o::Library library(&argc, &argv);
    if (argc != 3) {
        std::cout << "Usage: " << argv[0]
                  << " <input mesh> <interpolation radius>\n";
        return -1;
    }
    std::string input_mesh = argv[1];
    o::Real interpolation_radius = std::stod(argv[2]);
    printf("Input mesh: %s\n", input_mesh.c_str());
    printf("Interpolation radius: %f\n", interpolation_radius);
    o::Mesh mesh;
    if (input_mesh == std::string("internal_box")) {
        mesh = o::build_box(library.world(), OMEGA_H_SIMPLEX, 1, 1, 1, 100, 100,
                            0, false);
    } else {
        mesh = o::binary::read(input_mesh, library.self());
    }
    set_global_tag(mesh);
    o::Read<o::Real> data(mesh.nverts(), 0.0);
    o::Reals data_face(mesh.nfaces(), 0.0);
    OMEGA_H_CHECK(mesh.has_tag(o::FACE, "global"));
    OMEGA_H_CHECK(mesh.has_tag(o::VERT, "global"));
    mesh.add_tag<o::Real>(0, "node_values_from_xgc", 1, data);
    mesh.add_tag<o::Real>(0, "node_values_from_degas2", 1, data);
    mesh.add_tag<o::Real>(2, "n_sq_from_degas2", 1, data_face);
    printf("Mesh loaded with %d elements\n", mesh.nelems());
    compute_cell_area_tag(mesh);

    MPI_Comm comm = MPI_COMM_WORLD;
    omega_h_coupler(comm, mesh, interpolation_radius);

    return 0;
}
