declare -a Envs=("N17E073" "N43W080" "N45W123" "N47W124")
declare -a Strategies=("random")
declare -a Configs=(
"h_2" "h_5" "h_7" "h_11" "h_13" "h_17"
"m_2" "m_3" "m_5" "m_7" "m_11" "m_17"
"min_0.005" "min_0.01" "min_0.05" "min_0.1" "min_0.15" "min_0.2"
"max_0.2" "max_0.3" "max_0.5" "max_0.7" "max_0.9" "max_1.1"
)

for seed in {0..0}; do
  for env in ${Envs[@]}; do
    for strategy in ${Strategies[@]}; do
      for config in ${Configs[@]}; do
        echo $seed $env $strategy $config 
        mkdir -p ./loginfo/$seed/$env/$strategy/
        python main.py \
          --config ./configs/$config.yaml \
          --env-name $env \
          --strategy $strategy \
          --seed $seed \
          > "./loginfo/${seed}/${env}/${strategy}/${config}.txt"
        done
      done
    done
  done
