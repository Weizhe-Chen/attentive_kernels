declare -a Envs=("N17E073" "N43W080" "N45W123" "N47W124")
declare -a Strategies=("random" "active" "myopic")
declare -a Configs=("rbf" "gibbs" "dkl" "ak")

mkdir -p ./loginfo

for seed in {0..1}; do
  for env in ${Envs[@]}; do
    for strategy in ${Strategies[@]}; do
      for config in ${Configs[@]}; do
        echo $seed $env $strategy $config 
        python main.py --config ./configs/$config.yaml --env-name $env --strategy $strategy --seed $seed > ./loginfo/"${seed}_${env}_${strategy}_${config}".txt
      done
    done
  done
done
