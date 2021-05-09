# increase
* true: 加入新拓张的节点训练 false: 使用原始网络中某些点训练

# init
* true: 使用前一年份的作初始化, false: 重新训练

# strategy 
* strategy: incremental, retrain
    * incremental: detect, ewc, replay 都false时, 只训练新节点 ->lowerbound
    * retrain: detect, ewc, replay 都必须false, 所有节点上都训练, 目前是新init一个模型 -> upperbound

# detect
* detect: true,
* detect_strategy: original, feature 
    * original: 计算pre_data和cur_data的原始数据序列的分布的JS-div
    * feature: 计算pre_data和cur_data在输入pre_year的模型得到的feature的每一维JS-div的加总

# ewc
* ewc: false,
* ewc_strategy: ewc, 
    * ewc: 
    * l2: l2范数约束模型
* ewc_lambda: 1.0,

# replay
* replay: true,
* replay_strategy: random,
* replay_num_samples: 100