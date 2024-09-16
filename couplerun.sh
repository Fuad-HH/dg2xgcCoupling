#! /bin/bash
cd /lore/hasanm4/wsources/dg2xgcCoupling/
# delete if Run directory already exists
if [ -d Run ]; then rm -r Run; fi
mkdir Run
cd Run

declare -a PIDS=()

kill_stuff() {
  for PID in "${PIDS[@]}"
  do
    echo "Killing $PID"
    kill -9 $PID
  done
}


run_stuff() {
  mpirun -np 1 $1 /lore/hasanm4/wsources/pumipush/meshes/LCPPcoarse.osh &
  PIDS+=($!)
}
#mpirun -np 1 $1 /lore/hasanm4/wsources/pumipush/meshes/square.osh &

trap "kill_stuff" SIGINT

run_stuff ../build/dg2xgcCoupler 
run_stuff ../build/dummyxgc 
run_stuff ../build/dummydg2

wait
