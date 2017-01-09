#bash

num=10
name=simple


for i in `seq 1 1 $num`; do

echo $starting simulation run $i

time python -m langevin_model.simulation sim --name $name --steps 100000
time python -m run
python -m langevin_model.simulation next --name $name --start


done

