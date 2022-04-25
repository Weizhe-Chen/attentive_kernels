declare -a Envs=("N17E073" "N43W080" "N45W123" "N47W124")
declare -a Strategies=("random" "active" "myopic")
declare -a Kernels=("rbf" "gibbs" "dkl" "ak")

for seed in {0..9}; do
  for env in ${Envs[@]}; do
    for strategy in ${Strategies[@]}; do
      for kernel in ${Kernels[@]}; do
        echo $seed $env $strategy $kernel 
        mkdir -p ./loginfo/$seed/$env/$strategy/
        python main.py --config ./configs/$kernel.yaml --env-name $env --strategy $strategy --seed $seed > "./loginfo/${seed}/${env}/${strategy}/${kernel}.txt"
      done
    done
  done
done
