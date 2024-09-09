#include "node2cell.hpp"


int main(int argc, char* argv[])
{
    auto lib = o::Library(&argc, &argv);
    auto world = lib.world();
    auto nrank = world->size();
    auto rank = world->rank();

    if(argc != 2)
    {
        printf("Usage: %s <mesh_file>\n", argv[0]);
        return 1;
    }
    std::string mesh_file = argv[1];
    auto mesh = o::binary::read(mesh_file, lib.self());
    printf("Mesh loaded with %d elements\n", mesh.nelems());
    set_sinxcosy_tag(mesh);
    node_average2cell(mesh);
    render(mesh, 0, rank);

    return 0;
}