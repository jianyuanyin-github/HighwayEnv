# RL-and-MPC
Masterthesis code von Jianyuan Yin

# How to run
  # 激活虚拟环境
  source ~/forcespro38-env/bin/activate

  # 设置环境变量
  export PYTHONPATH="/home/yinjianyuan/forcespro":$PYTHONPATH

  # 安装RL依赖
  pip install -r requirements_rl.txt

  2. 训练RL智能体

  # 赛道单车环境训练
  python scripts/sb3_racetrack_single_ppo.py

  # 汇入环境训练
  python scripts/sb3_merge_PPO.py

  # 环岛环境训练
  python scripts/sb3_roundabout_ppo.py

  # 交叉路口环境训练
  python scripts/sb3_intersection_ppo.py

  3. 评估训练结果

  # 设置脚本中的TRAIN=False来评估模型
  # 模型会自动加载最新的训练结果进行测试

  4. 环境配置

  # 修改环境参数在相应的环境文件中：
  # highway_env/envs/racetrack_env_single.py - 赛道单车环境
  # highway_env/envs/merge_env.py - 汇入环境  
  # highway_env/envs/roundabout_env.py - 环岛环境
  # highway_env/envs/intersection_env.py - 交叉路口环境
