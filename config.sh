cmake -S . -B build \
    -DCMAKE_CXX_COMPILER=/lore/hasanm4/wsources/dg2xgcDeps/kokkos/bin/nvcc_wrapper \
    -DCMAKE_BUILD_TYPE=debug \
    -DBUILD_SHARED_LIBS=ON \
    -DOmega_h_ROOT=/lore/hasanm4/wsources/dg2xgcDeps/ohInstall \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
