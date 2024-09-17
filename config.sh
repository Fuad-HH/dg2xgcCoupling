export NVCC_WRAPPER_DEFAULT_COMPILER=`which mpicxx`
cmake -S . -B build \
    -DCMAKE_CXX_COMPILER=/lore/mersoj2/laces-software/build/ADA89/kokkos/install/bin/nvcc_wrapper \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_SHARED_LIBS=ON \
    -DOmega_h_ROOT=/lore/mersoj2/laces-software/build/ADA89/omega_h/install/ \
    -Dpcms_ROOT=/lore/hasanm4/wsources/dg2xgcDeps/build/pcms/ADA89/pcms/install \
    -Dperfstubs_DIR=/lore/mersoj2/laces-software/build/perfstubs/install/lib64/cmake/ \
    -Dredev_DIR=/lore/mersoj2/laces-software/build/ADA89/redev/install/lib64/cmake/redev/ \
    -DKokkos_ROOT=/lore/mersoj2/laces-software/build/ADA89/kokkos/install/ \
    -DCatch2_ROOT=/lore/mersoj2/laces-software/build/Catch2/install \
    -Dinterpolator_DIR=/lore/hasanm4/wsources/dg2xgcDeps/build/ADA89/interpolator/lib/cmake/interpolator/ \
    -DCMAKE_PREFIX_PATH=/lore/hasanm4/wsources/dg2xgcDeps/build/ADA89/interpolator/ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
