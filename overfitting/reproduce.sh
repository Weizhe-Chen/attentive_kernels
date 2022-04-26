declare -a Envs=("helens" "N17E073" "N43W080" "N45W123" "N47W124")
declare -a Configs=("rbf" "gibbs" "dkl" "ak")

for env in ${Envs[@]}; do
  for config in ${Configs[@]}; do
    echo $env $config 
    python main.py --config ./configs/$config.yaml --env-name $env --num-train 300
  done
done
