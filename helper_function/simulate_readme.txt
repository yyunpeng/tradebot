1. 首先 import 东西

from helper_function.simulate import Simulate

2. 然后创建一个 Simulate class 的 object，创建的时候一定要把 assetPrices 放进 parameter 里

simulator = Simulate(assetPrices_perMinute)

3. 可以运用 simulator.execute() 的 function 来做一次的 simulation

dimension, mean_matrix_list, covariance_matrix_list = simulator.execute(t=30, r=0.1, path_num=100)

Input：

- t: 意思就是从哪一个 Tick_Index 开始 simulate
- tau_limit: 意思是从 t 开始要做多少个 simulation。最终会做多少个 simulation 是看 tau 这个 variable 的数值，而 tau 的计算公式是 tau = min(tau_limit, 239-t)
  tau_limit 预设值是 30，代表 tau 的预设值是 30。
- wait_time： 是指要等开市后多久才可以做第一次的 simulation。为啥要等是因为你需要前面的等待时间才有足够的数据去建立一个 Sigma Head 的 matrix (sample covariance matrix).  
  预设值是 1800 代表开市后的 30 分钟 (10:00:00 am)
- r： 利率，根据上一次的聊天，这利率代表中国投资者期望通过中国境内最优质投资 (相当于美国的政府国债) 拿到的每年回报率。
  预设值是 0.1 (代表 10% 利率)
- path_num: 每只股票你想做多少条 path 的 simulation，预设值是 100 (如果 path_num=100，平均每次 simulation 一个 timestamp 都需要花 0.15 秒做运算)
- delta_t: 是 GBM 公式里头的 delta T。预设值是 1/(252*14400)，代表每一秒钟的 delta time

Output：

- dim: dim = tau+1，就是要做多少个 simulation 包括现在这一分钟 (不需要 simulate) 的数值。它代表 mean_matrix_list 里头每一个 mean matrix 和 
  covariance_matrix_list 里头每一个 covariance matrix 的 dimension。
  dim 一般的数值是 31，因为是 tau=30 加上自己。
- mean_matrix_list: 一个 list 长度是 100。List 里头每个 element 代表一只股票，而且是一个 (tau, ) 长度的 one-dimensional NumPy array。
  代表 simulate 之后，每一只股票在每个 timestamp 在 100 条 path 的平均价格 (模拟价格，非真实价格)。
- covariance_matrix_list: 一个 list 长度是 100。List 里头每个 element 代表一只股票，而且是一个 covariance matrix in (dim X dim) dimension。
  它代表每只股票在每一个 timestamp 的 covariance。


