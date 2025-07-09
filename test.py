import numpy as np
from scipy.stats import norm
import time


# --- 假设的外部函数和参数 ---
# 您需要根据实际问题定义这些函数和变量
# 为了使代码可运行，这里提供一个简单的占位符实现

# f_xt: 状态转移函数，根据 t 时刻之前的粒子状态，预测 t 时刻的粒子状态。
# 在 MATLAB 示例中，它体现为 particle = particle + v_meas * dt;
# 这里假设 f_xt(particles, t_step) = particles + some_velocity * dt
# 为了匹配MATLAB的v_meas，这里简化为不依赖t_step的简单加法
# 假设状态是二维的 [x; y]
def f_xt(particles, velocity_measurement, dt_step):
    """
    状态转移函数：根据运动模型预测粒子。
    Args:
        particles (np.ndarray): 形状为 (状态维度, 粒子数) 的粒子数组。
        velocity_measurement (np.ndarray): 形状为 (状态维度,) 的速度测量向量。
        dt_step (float): 时间步长。
    Returns:
        np.ndarray: 预测后的粒子。
    """
    # 简单的线性运动模型
    return particles + velocity_measurement[:, np.newaxis] * dt_step


# h_zt: 观测函数，根据粒子状态预测观测值。
# 在 MATLAB 示例中，它体现为 z_est = DEM_height(particle, DEM);
# 假设 DEM_height(position) 是一个返回高度的函数
# 这里假设状态是二维位置，观测是一维的高度
def h_zt(particles_state, DEM_model):
    """
    观测函数：根据粒子状态预测观测值（例如高度）。
    Args:
        particles_state (np.ndarray): 形状为 (状态维度, 粒子数) 的粒子数组。
        DEM_model: 数字高程模型或其他用于预测观测值的模型。
    Returns:
        np.ndarray: 形状为 (粒子数,) 的预测观测值。
    """
    # 占位符：实际应根据DEM模型计算高度
    # 假设 DEM_model 能够从二维位置提取高度
    # 示例：简单的二维位置到一维高度的映射
    # return DEM_model.get_height(particles_state[0,:], particles_state[1,:])
    return (particles_state[0, :] ** 2 + particles_state[1, :] ** 2) / 100.0  # 示例：一个简单的二次曲面


# likelihood: 似然函数
# 在 MATLAB 示例中，它被显式定义。这里使用 scipy.stats.norm.pdf
def compute_likelihood(predicted_measurement, actual_measurement, measurement_noise_variance):
    """
    计算似然值。
    Args:
        predicted_measurement (np.ndarray): 预测的观测值。
        actual_measurement (float): 实际观测值。
        measurement_noise_variance (float): 观测噪声方差 R。
    Returns:
        np.ndarray: 似然值数组。
    """
    return norm.pdf(actual_measurement, loc=predicted_measurement,
                    scale=np.sqrt(measurement_noise_variance)) + 1e-300  # 添加一个极小值防止0


# Resample: 重采样函数
def Resample(particles, weights):
    """
    系统重采样（Systematic Resampling）
    Args:
        particles (np.ndarray): 形状为 (状态维度, 粒子数) 的粒子数组。
        weights (np.ndarray): 形状为 (粒子数,) 的权重数组。
    Returns:
        tuple: (重采样后的粒子, 重采样后的权重)
    """
    N = particles.shape[1]
    # 确保权重归一化
    weights /= np.sum(weights)

    # 累积分布函数
    cumulative_weights = np.cumsum(weights)
    # 随机起点
    r0 = np.random.uniform(0, 1 / N)
    # 均匀分布的随机数，用于选择粒子
    r = r0 + np.arange(N) / N

    # 查找对应的粒子索引
    indices = np.searchsorted(cumulative_weights, r)

    # 重采样粒子
    particles_resampled = particles[:, indices]
    # 重置权重
    weights_resampled = np.ones(N) / N

    return particles_resampled, weights_resampled


# --- APF 函数主体 ---

def APF(y_obs, v_meas_seq, N, Q, R, x0, dt, DEM_model, prior_std=2.0):
    """
    辅助粒子滤波（Auxiliary Particle Filter）的Python实现，参照MATLAB逻辑。

    Args:
        y_obs (np.ndarray):       观测序列 (长度 T, 观测维度)。
        v_meas_seq (np.ndarray):  速度测量序列 (形状为 (T-1, 状态维度))，对应MATLAB的v_meas。
        N (int):                  粒子数。
        Q (np.ndarray):           过程噪声协方差矩阵 (形状为 (状态维度, 状态维度))。
        R (float):                观测噪声方差 (假设观测是一维)。
        x0 (np.ndarray):          初始状态均值 (形状为 (状态维度, ))。
        dt (float):               时间步长。
        DEM_model:                数字高程模型或其他用于预测观测值的模型。
        prior_std (float):        初始粒子标准差。

    Returns:
        x_est (np.ndarray):       形状为 (T, 状态维度) 的状态估计 (粒子加权均值)。
        ESS (np.ndarray):         长度 T 的有效样本数列表。
        runtime (float):          总耗时 (秒)。
        cov_increase_count (np.ndarray): 记录协方差增大的次数。
    """
    T = y_obs.shape[0]  # 观测序列的长度
    state_dim = x0.shape[0]  # 状态维度

    # 初始化粒子：围绕初始均值采样，并添加噪声
    # 形状为 (状态维度, 粒子数)
    particles = x0[:, np.newaxis] + np.random.normal(0, prior_std, size=(state_dim, N))
    weights = np.ones(N) / N  # 初始化粒子权重

    x_est = np.zeros((T, state_dim))  # 存储状态估计
    ESS = np.zeros(T)  # 有效样本数

    # 协方差分析相关
    cov_increase_count = np.zeros(T)  # 记录协方差增大的时间步

    start_time = time.time()  # 开始计时

    # 第一个时间步的初始估计
    x_est[0, :] = np.sum(particles * weights, axis=1)  # 初始粒子集的加权平均作为第一个估计值

    for t_step in range(1, T):  # 从第二个时间步 (k=2) 开始，对应Python的索引 1
        # --- MATLAB 逻辑中的 'auxiliary' 阶段 ---
        # 1. 粒子预测（根据运动模型）
        # 这里使用上一个时间步的速度测量 v_meas_seq[t_step - 1]
        particles_pred_aux = f_xt(particles, v_meas_seq[t_step - 1], dt)

        # 2. 计算辅助权重
        # 估计每个粒子对应的测量值
        z_est_aux = h_zt(particles_pred_aux, DEM_model)
        # 根据预测测量值与实际测量值的匹配程度更新权重（这里是辅助似然）
        aux_lik = compute_likelihood(z_est_aux, y_obs[t_step], R)  # 使用当前时间步的观测 y_obs[t_step]

        # 辅助权重： w̃ ∝ w_{t-1}^{(i)} * p(y_t | f(x_{t-1}^{(i)}))
        aux_weights = weights * aux_lik
        aux_weights += 1e-300  # 防止除以零或数值问题
        aux_weights /= np.sum(aux_weights)  # 归一化辅助权重

        # 3. 从辅助分布中重采样粒子索引 (index)
        # 根据辅助权重 indices 选择粒子
        indices = np.random.choice(N, size=N, replace=True, p=aux_weights)
        # 重采样粒子，并重置权重
        particles, weights = Resample(particles, aux_weights)  # 注意这里用辅助权重重采样，并重置权重为1/N

        # --- MATLAB 逻辑中的 'time update' 阶段 ---
        # 4. 对重采样后的粒子添加过程噪声（扩散）
        # 这里应用f_xt，但如果f_xt是线性增量，则过程噪声直接加在粒子上
        # 这里的 f_xt 应该已经被辅助阶段使用了，所以这里是直接添加过程噪声
        # MATLAB 中的 `particle = particle + sig_proc*randn(2,numParticle)*dt;` 对应于这里
        particles += np.random.multivariate_normal(np.zeros(state_dim), Q, size=N).T * dt  # 乘以dt是为了匹配MATLAB的dt

        # --- MATLAB 逻辑中的 'measurement update' 阶段 ---
        # 5. 再次计算每个粒子的测量似然，用于最终权重更新
        # 估计每个粒子当前的测量值
        z_est_final = h_zt(particles, DEM_model)
        # 计算最终权重：w_t^{(i)} ∝ p(y_t | x_t^{(i)})
        final_lik = compute_likelihood(z_est_final, y_obs[t_step], R)

        weights = final_lik  # 最终权重直接是似然，因为前一步重采样已经将权重重置为1/N (隐式)
        weights += 1e-300
        weights /= np.sum(weights)  # 归一化最终权重

        # --- 估计 & ESS ---
        # 计算当前时间步的状态估计（加权平均）
        x_est[t_step, :] = np.sum(particles * weights, axis=1)
        # 计算有效样本数 (ESS)
        ESS[t_step] = 1.0 / np.sum(weights ** 2)

        # --- MATLAB 逻辑中的 'cov test' 阶段 ---
        # 这部分在MATLAB中是用于分析的，这里也加上
        # 为了计算 dcov_mi (预测后的协方差)，我们需要在添加过程噪声之后，进行测量更新之前
        # 但在APF中，这里的协方差通常指最终加权粒子集的协方差
        # 这里为了模拟MATLAB的逻辑，我们计算重采样后的粒子集的协方差

        # 再次重采样一次用于协方差分析（匹配MATLAB的 `[particle_t, weight_t] = Resample(particle, weight);`）
        particles_for_cov_test, _ = Resample(particles, weights)

        # 注意：MATLAB的 `dcov_mi` 是在第一次辅助权重更新前后的粒子协方差，
        # 这里的 `dcov_pl` 对应MATLAB中的 `dcov_pl`，即最终重采样后的粒子协方差。
        # 如果要精确匹配MATLAB的 `dcov_mi`，需要在 `particles_pred_aux` 计算后，重采样前进行。
        # 但MATLAB的这个协方差判断条件 `if (dcov_pl > dcov_mi)` 在APF中通常用于诊断。

        # 计算当前粒子集的协方差行列式
        # 注意：这里是基于最终加权粒子集的协方差，如果MATLAB的 dcov_mi 指的是预测协方差，则需要调整计算位置。
        # 这里我们假定 dcov_pl 是最终加权粒子集的协方差
        current_cov_det = np.linalg.det(np.cov(particles_for_cov_test))

        # MATLAB的 covIncrease 逻辑通常用于比较预测后的协方差与测量更新后的协方差
        # 在这里，我们简化为比较当前最终粒子集的协方差，如果其相比“某种先验”增大了
        # 为了更直接地匹配MATLAB，假设 dcov_mi 是辅助阶段预测后的粒子协方差
        # 这里重新计算一个近似的 dcov_mi
        # dcov_mi_approx = np.linalg.det(np.cov(particles_pred_aux))
        # if current_cov_det > dcov_mi_approx:
        #     cov_increase_count[t_step] += 1

        # 为了严格模拟MATLAB的 `dcov_mi` 计算点，我们需要在 `particles_pred_aux` 之后和重采样之前计算
        # 但APF的重采样是基于辅助权重，所以协方差的“先验”和“后验”定义会有点微妙。
        # 在这里，我们保留MATLAB的判断逻辑，但可能需要对 `dcov_mi` 的计算点进行更精确的定义
        # 考虑到当前结构，我们假设 dcov_mi 应该是在辅助阶段预测后，未进行第二次测量更新前的粒子分布。
        # MATLAB中的 `dcov_mi = det(cov(particle'))` 在 'measurement update' 之前。
        # 也就是在 `particles` 经过 `+ sig_proc*randn` 之后。
        # 为了匹配，我们在这里再计算一个“先验协方差”

        # 这里为了匹配MATLAB的 `dcov_mi`，我们需要在 `particles` 经过过程噪声扩散后计算一次
        # 由于我们无法在Python中直接获取MATLAB中 `dcov_mi` 的精确计算点，
        # 我们这里就简化为一个后验协方差的比较，或者可以移除此诊断，除非有明确定义
        # 为保持与MATLAB代码的一致性，我们将保留这个逻辑，但请注意其精确含义可能需要进一步上下文

        # 模拟MATLAB的dcov_mi, 发生在time update之后，measurement update之前
        dcov_mi_matlab_equivalent = np.linalg.det(np.cov(particles))
        # 模拟MATLAB的dcov_pl, 发生在所有更新和估计之后，额外一次Resample
        dcov_pl_matlab_equivalent = np.linalg.det(np.cov(particles_for_cov_test))

        # 存储协方差分析结果
        # dcov_res[:, k, i] = covAnal(cov(particle_t'));
        # 这里需要您提供 `covAnal` 函数的实现
        # dcov_res[t_step, :] = covAnal(np.cov(particles_for_cov_test))

        if (dcov_pl_matlab_equivalent > dcov_mi_matlab_equivalent):
            cov_increase_count[t_step] += 1

        # --- 可选的再次重采样（ESS 太小） ---
        if ESS[t_step] < N / 2:
            # print(f"Resampling due to low ESS at time step {t_step}. ESS: {ESS[t_step]:.2f}")
            particles, weights = Resample(particles, weights)  # 再次重采样，重置权重为1/N

    runtime = time.time() - start_time
    return x_est, ESS, runtime, cov_increase_count


# --- 示例使用 (需要您替换为实际数据和模型) ---
if __name__ == "__main__":
    # 模拟数据生成（对应 MATLAB 场景）
    T = 100  # 时间步数
    state_dim = 2  # 状态维度 (x, y)
    N_particles = 1000  # 粒子数

    # 过程噪声协方差矩阵 (对应 MATLAB 的 sig_proc^2)
    # 假设状态是独立的，或者是一个简单的二维高斯噪声
    Q_proc_std = 0.1  # MATLAB sig_proc
    Q = np.eye(state_dim) * (Q_proc_std ** 2)

    # 观测噪声方差 (对应 MATLAB 的 sig_meas^2)
    R_meas_std = 0.5  # MATLAB sig_meas
    R = R_meas_std ** 2

    x0_true = np.array([0.0, 0.0])  # 真实初始状态
    x0_init_err = np.array([1.0, -0.5])  # 初始估计误差
    x0_est = x0_true + x0_init_err  # 初始估计值

    prior_std = 2.0  # 初始粒子标准差
    dt = 1.0  # 时间步长

    # 模拟真实轨迹 (x_true) 和速度测量 (v_meas_seq)
    # 为了简化，我们假设一个简单的线性运动，并加上噪声
    x_true_trajectory = np.zeros((T, state_dim))
    v_meas_sequence = np.zeros((T, state_dim))  # 速度测量序列，对应MATLAB的v_meas
    z_true_measurements = np.zeros(T)  # 真实观测值

    # 简单的运动模型和观测模型
    # 在真实应用中，这些会从传感器或环境模型中获取
    current_true_x = x0_true
    for i in range(T):
        x_true_trajectory[i, :] = current_true_x
        # 模拟速度测量
        if i < T - 1:
            # 假设一个恒定速度加上一些测量噪声
            true_velocity = np.array([0.1, 0.2])
            v_meas_sequence[i, :] = true_velocity + np.random.normal(0, 0.05, size=state_dim)  # 加上速度测量噪声

        # 模拟真实观测
        z_true_measurements[i] = h_zt(current_true_x[:, np.newaxis], DEM_model=None)[0]

        z_true_measurements[i] += np.random.normal(0, R_meas_std)  # 加上观测噪声

        # 更新真实状态
        current_true_x = current_true_x + true_velocity * dt + np.random.normal(0, Q_proc_std, size=state_dim)  # 真实过程噪声

    # 运行APF
    # 注意：这里的 y_obs 应该是模拟的带噪声的观测 z_meas
    # DEM_model 需要被实际传入
    estimated_states, ess_values, total_runtime, cov_increase_counts = APF(
        y_obs=z_true_measurements,
        v_meas_seq=v_meas_sequence,
        N=N_particles,
        Q=Q,
        R=R,
        x0=x0_est,  # 传入初始估计值
        dt=dt,
        DEM_model=None,  # 您需要提供一个实际的 DEM_model
        prior_std=prior_std
    )

    print(f"APF 运行完成。总耗时：{total_runtime:.4f} 秒")
    print(f"最终估计状态均值：{estimated_states[-1, :]}")
    print(f"平均有效样本数 (ESS)：{np.mean(ess_values):.2f}")
    print(f"协方差增大的时间步数量：{np.sum(cov_increase_counts)}")

    # 简单绘图（如果需要安装 matplotlib）
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(x_true_trajectory[:, 0], x_true_trajectory[:, 1], 'r-', label='True Path')
        plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'b--', label='Estimated Path')
        plt.scatter(x0_est[0], x0_est[1], color='green', marker='o', label='Initial Est')
        plt.title('2D Path Estimation')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(ess_values)
        plt.title('Effective Sample Size (ESS)')
        plt.xlabel('Time Step')
        plt.ylabel('ESS')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not installed. Skipping plot.")