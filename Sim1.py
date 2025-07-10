import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.stats import norm


# ======================
# 1. 公共工具函数
# ======================
@njit
def f_xt(x, t):
    """Kitagawa 模型的状态转移函数"""
    return 0.5 * x + 25 * x / (1 + x**2) + 8 * np.cos(1.2 * t)

@njit
def h_zt(x):
    """Kitagawa 模型的观测函数"""
    return 0.05 * x**2

@njit(parallel=True)
def compute_likelihood(candidates, y_obs_t, R):
    n = candidates.shape[0]
    lik = np.empty(n)
    sqrt_R = np.sqrt(R)
    for i in prange(n):
        h_val = h_zt(candidates[i])  # 如果 h(x)=x，否则替换这里
        lik[i] = (1.0 / (np.sqrt(2 * np.pi) * sqrt_R)) * np.exp(-0.5 * ((y_obs_t - h_val) / sqrt_R) ** 2)
    return lik

def simulate_kitagawa(T, Q, R, x0=5.0, seed=None):
    """
    模拟 Kitagawa 一维系统，返回长度 T 的真实状态与观测序列
    状态：x_t = f(x_{t-1}, t) + w_t,  w_t ~ N(0, Q)
    观测：y_t = h(x_t) + v_t,       v_t ~ N(0, R)
    """
    if seed is not None:
        np.random.seed(seed)
    x_true = np.zeros(T)
    y_obs = np.zeros(T)
    x_prev = x0
    for t in range(1, T+1):
        x_curr = f_xt(x_prev, t) + np.random.normal(0, np.sqrt(Q))
        y_curr = h_zt(x_curr) + np.random.normal(0, np.sqrt(R))
        x_true[t-1] = x_curr
        y_obs[t-1] = y_curr
        x_prev = x_curr
    return x_true, y_obs

# ======================
# 2. Bootstrap SIR‑PF 实现
# ======================
def SIR_PF(y_obs, N, Q, R, x0, prior_std=2.0):
    """
    标准 Bootstrap 粒子滤波（SIR-PF）
    Args:
        y_obs:       观测序列 (长度 T)
        N:           粒子数
        Q, R:        过程噪声方差, 观测噪声方差
        x0:          初始状态均值
        prior_std:   初始粒子标准差
    Returns:
        x_est:       长度 T 的状态估计 (粒子均值)
        ESS:         长度 T 的有效样本数列表
        runtime:     总耗时 (秒)
    """
    T = len(y_obs)
    # 初始化
    particles = np.random.normal(x0, prior_std, size=N)  # N 颗 t=0 粒子
    weights = np.ones(N) / N
    x_est = np.zeros(T)
    ESS = np.zeros(T)

    start = time.time()
    for t in range(1, T+1):
        # 1) 预测：从 p(x_t | x_{t-1}) 采样
        particles = f_xt(particles, t) + np.random.normal(0, np.sqrt(Q), size=N)

        # 2) 更新权重：w ∝ p(y_t | x_t^{(i)})
        # lik = norm.pdf(y_obs[t-1], loc=h_zt(particles), scale=np.sqrt(R))
        lik = compute_likelihood(particles, y_obs[t - 1], R)
        weights = weights * lik
        weights += 1e-300  # 防止全为零
        weights /= np.sum(weights)

        # 3) 计算 ESS
        ESS[t-1] = 1.0 / np.sum(weights**2)

        # 4) 重采样 (SIR)：按当前权重重新均匀抽取 N 个粒子
        idx = np.random.choice(N, size=N, replace=True, p=weights)
        particles = particles[idx]
        weights.fill(1.0 / N)

        # 5) 估计：取粒子均值
        x_est[t-1] = np.mean(particles)
    runtime = time.time() - start

    return x_est, ESS, runtime

# ======================
# 3. EKF‑PF (EPF) 简化实现
# ======================
def EKF_PF(y_obs, N, Q, R, x0, P0=1.0):
    """
    扩展卡尔曼粒子滤波 (EKF-PF) 单维简单版：
    每个粒子使用 EKF 产生一个高斯 proposal，再重新计算权重并重采样。
    Args:
        y_obs:   观测序列 (长度 T)
        N:       粒子数
        Q, R:    过程和观测噪声方差
        x0:      初始状态均值
        P0:      初始状态协方差
    Returns:
        x_est:   长度 T 的状态估计
        ESS:     有效样本数列表
        runtime: 运行时间
    """
    T = len(y_obs)
    # Jacobians: dx f/dx 时刻依赖 x, dx h/dx 依赖 x
    def df_dx(x, t):
        # f(x,t) = 0.5 x + 25 x/(1+x^2) + 8 cos(1.2 t)
        return 0.5 + 25*(1 - x**2)/(1 + x**2)**2

    def dh_dx(x):
        # h(x) = 0.05 x^2
        return 0.1 * x

    # 初始化 N 个粒子及其协方差矩阵
    particles = np.random.normal(x0, np.sqrt(P0), size=N)
    P_particles = np.ones(N) * P0
    weights = np.ones(N) / N

    x_est = np.zeros(T)
    ESS = np.zeros(T)

    start = time.time()
    for t in range(1, T+1):
        # 每个粒子生成一个 proposal（EKF 更新）
        proposals = np.zeros(N)
        proposal_vars = np.zeros(N)
        for i in range(N):
            x_prev = particles[i]
            P_prev = P_particles[i]

            # 1) 预测步 (EKF)
            A = df_dx(x_prev, t)                 # 雅可比
            x_pred = f_xt(x_prev, t)             # m_{t|t-1}
            P_pred = A * P_prev * A + Q         # P_{t|t-1}

            # 2) 更新步 (EKF)
            H = dh_dx(x_pred)
            if abs(H) < 1e-6:  # 防止过小
                H = 1e-6
            Kk = P_pred * H / (H * P_pred * H + R)  # 卡尔曼增益
            x_upd = x_pred + Kk * (y_obs[t-1] - h_zt(x_pred))
            P_upd = (1 - Kk*H) * P_pred

            # 将 EKF 更新结果做 Proposal 分布 N(x_upd, P_upd)
            proposals[i] = np.random.normal(x_upd, np.sqrt(max(P_upd,1e-8)))
            proposal_vars[i] = P_upd

            # 缓存新的 P_upd 供下一步使用
            P_particles[i] = P_upd

        # 3) 计算重要性权重 w ∝ p(y|proposals)*p(proposals|particles_prev) / q(proposals)
        #   其中 q(...) = N(proposals; x_upd, P_upd)
        #   而 p(proposals | x_{t-1}) = N(proposals; f(x_{t-1}), Q)
        #   因此： w_i ∝ N(y_t; h(proposals_i), R) * N(proposals_i; f(x_{t-1}),Q) /
        #                    N(proposals_i; x_upd, P_upd)
        #   计算时取对数再 exponent 以避免数值下溢
        log_w = np.zeros(N)
        for i in range(N):
            # 观测似然
            log_lik = norm.logpdf(y_obs[t-1], loc=h_zt(proposals[i]), scale=np.sqrt(R))
            # 先验转移概率（过程模型）
            log_prior_trans = norm.logpdf(proposals[i], loc=f_xt(particles[i], t), scale=np.sqrt(Q))
            # Proposal 概率（EKF-Gaussian）
            log_q = norm.logpdf(proposals[i], loc=proposals[i], scale=np.sqrt(proposal_vars[i]))
            # 上式中 N(proposals; x_upd, P_upd) 近似即：
            #     norm.logpdf(proposals[i], loc=x_upd, scale=sqrt(P_upd))
            # 由于 proposals[i] 从该分布采样，故 loc=proposals[i]、scale=sqrt(proposal_vars[i]) 可近似
            log_w[i] = log_lik + log_prior_trans - log_q

        # 数值稳定：减去 max
        c = np.max(log_w)
        weights = np.exp(log_w - c)
        weights += 1e-300
        weights /= np.sum(weights)

        # 4) 计算 ESS
        ESS[t-1] = 1.0 / np.sum(weights**2)

        # 5) 重采样
        idx = np.random.choice(N, size=N, replace=True, p=weights)
        particles = proposals[idx]
        P_particles = P_particles[idx]
        weights.fill(1.0 / N)

        # 6) 估计
        x_est[t-1] = np.mean(particles)
    runtime = time.time() - start

    return x_est, ESS, runtime

# ======================
# 4. UPF 简单实现（UKF-PF 单维）
# ======================
def UPF(y_obs, N, Q, R, x0, P0=1.0, kappa=0.0):
    """
    单维 UPF (Unscented Particle Filter) 实现：
    每颗粒子在预测-更新时用 Unscented 变换得到 proposal。
    Args:
        y_obs:   观测序列
        N:       粒子数
        Q, R:    噪声方差
        x0:      初始均值
        P0:      初始方差
        kappa:   UKF 参数 (n=1 时可设为 0)
    Returns:
        x_est:   长度 T 的状态估计
        ESS:     有效样本数
        runtime: 运行时间
    """
    T = len(y_obs)
    n = 1
    lam = kappa  # 对单维可简单设 lam = 0

    # 计算 UKF 2n+1 个 sigma 点及权重
    def compute_sigma_points(xm, Pm):
        # 单维：sigma 点 = [ xm; xm + sqrt((n+lam)*Pm); xm - sqrt((n+lam)*Pm ) ]
        cov_sqrt = np.sqrt((n+lam)*Pm)
        return np.array([xm, xm+cov_sqrt, xm-cov_sqrt])

    def compute_weights():
        # 单维的 2n+1 权重
        Wm = np.array([lam/(n+lam), 1/(2*(n+lam)), 1/(2*(n+lam))])
        Wc = Wm.copy()
        return Wm, Wc

    Wm, Wc = compute_weights()

    # 初始化
    particles = np.random.normal(x0, np.sqrt(P0), size=N)
    P_particles = np.ones(N) * P0
    weights = np.ones(N) / N

    x_est = np.zeros(T)
    ESS = np.zeros(T)

    start = time.time()
    for t in range(1, T+1):
        proposals = np.zeros(N)
        proposal_vars = np.zeros(N)

        # 对每个粒子，应用 UKF 得到 Proposal 分布：N(x_t|t^{(i)}, P_t|t^{(i)})
        for i in range(N):
            x_prev = particles[i]
            P_prev = P_particles[i]

            # —— Unscented 预测 ——
            # 1) 构造 sigma 点
            sigmas = compute_sigma_points(x_prev, P_prev)  # shape = (3,)
            # 2) 通过 f_xt 逐点传播
            sigmas_pred = f_xt(sigmas, t)
            # 3) 计算预测均值与方差
            x_pred = np.dot(Wm, sigmas_pred)
            P_pred = np.dot(Wc, (sigmas_pred - x_pred)**2) + Q

            # —— Unscented 更新 ——
            # 1) 构造预测 sigma
            sigmas2 = compute_sigma_points(x_pred, P_pred)
            # 2) 通过 h_zt 计算观测 sigma
            sigmas_obs = h_zt(sigmas2)
            # 3) 观测均值与协方差
            y_pred = np.dot(Wm, sigmas_obs)
            P_yy = np.dot(Wc, (sigmas_obs - y_pred)**2) + R
            P_xy = np.dot(Wc, (sigmas2 - x_pred)*(sigmas_obs - y_pred))

            # 4) 卡尔曼增益、更新均值与方差
            Kk = P_xy / P_yy
            x_upd = x_pred + Kk * (y_obs[t-1] - y_pred)
            P_upd = P_pred - Kk * P_yy * Kk

            # 5) 将 UKF 更新结果当作 Proposal：N(x_upd, P_upd)
            proposals[i] = np.random.normal(x_upd, np.sqrt(max(P_upd,1e-8)))
            proposal_vars[i] = P_upd

            # 缓存 P_upd
            P_particles[i] = P_upd

        # 6) 计算重要性权重：同 EKF-PF 部分
        log_w = np.zeros(N)
        for i in range(N):
            log_lik = norm.logpdf(y_obs[t-1], loc=h_zt(proposals[i]), scale=np.sqrt(R))
            log_prior_trans = norm.logpdf(proposals[i], loc=f_xt(particles[i], t), scale=np.sqrt(Q))
            # Proposal 近似 N(x_upd, P_upd)
            log_q = norm.logpdf(proposals[i], loc=proposals[i], scale=np.sqrt(proposal_vars[i]))
            log_w[i] = log_lik + log_prior_trans - log_q

        c = np.max(log_w)
        weights = np.exp(log_w - c)
        weights += 1e-300
        weights /= np.sum(weights)

        # 7) 计算 ESS
        ESS[t-1] = 1.0 / np.sum(weights**2)

        # 8) 重采样
        idx = np.random.choice(N, size=N, replace=True, p=weights)
        particles = proposals[idx]
        P_particles = P_particles[idx]
        weights.fill(1.0 / N)

        # 9) 状态估计
        x_est[t-1] = np.mean(particles)

    runtime = time.time() - start
    return x_est, ESS, runtime

# ======================
# 5. BCPS‑PF 实现模块
# ======================
def BCPS_PF(y_obs, N, Q, R, x0, alpha=0.9, prior_std=2.0, max_batches=500):
    """
    Batch‑Cyclic Posterior Selection PF (BCPS‑PF)
    仅针对一维 Kitagawa 模型，返回粒子均值估计与所用批次数
    """
    T = len(y_obs)
    # 初始化粒子
    particles_prev = np.random.normal(x0, prior_std, size=N)
    x_est = np.zeros(T)
    K_list = np.zeros(T, dtype=int)
    Nt= N  # 初始粒子数
    accept_rate_list= np.zeros(T)

    start = time.time()
    for t in range(1, T+1):
        A_t = []

        # 无噪声预测值
        base_particles = f_xt(particles_prev, t)

        # 不断迭代“批次”直到接受粒子 ≥ alpha*N 或 批次数到达 max_batches
        for k in range(1, max_batches+1):
            # 1) 生成 N 个候选 = base + 过程噪声
            noise = np.random.normal(0, np.sqrt(Q), size=Nt)
            # print(noise.shape, base_particles.shape, noise.shape, Nt, N, k, t)
            candidates = base_particles + noise

            # 2) 计算当前批次 N 个候选的观测似然
            # lik = norm.pdf(y_obs[t-1], loc=h_zt(candidates), scale=np.sqrt(R)) #
            lik = compute_likelihood(candidates, y_obs[t - 1], R)
            # Lmax = np.max(lik)
            # Lmax = norm.pdf(0, 0, scale=np.sqrt(R))  # 基准似然
            Lmax = 0.3989422804014327 # 1/sqrt(2*pi) 观测噪声的标准正态分布值

            # 3) 接受概率 = lik/Lmax
            r = np.random.rand(Nt)
            accepted = candidates[r < (lik / Lmax)] # 接受的粒子
            # 补充一段代码，当 accepted 为空时，将lik中最大值和次大值对应的粒子添加到 accepted 中
            if len(accepted) == 0:
                max_lik_idx = np.argmax(lik)
                second_max_lik_idx = np.argsort(lik)[-2]
                accepted = np.array([candidates[max_lik_idx], candidates[second_max_lik_idx]])
            A_t.extend(accepted.tolist())

            if len(A_t) >= alpha * N:
                # print(len(A_t), "particles accepted int batch", k, "at time", t,)
                #计算接受率：最终获得的粒子数 / 用到的粒子数
                accept_rate = len(A_t) / (Nt * k)
                Nt = len(A_t)  # 更新当前批次粒子数
                break
            if k==max_batches:
                accept_rate = len(A_t) / (Nt * k)
                Nt = len(A_t)  # 达到最大批次数，停止采样
                # print(len(A_t), "达到最大值", k, "at time", t,)
                # print(lik / Lmax, "接受概率")
        accept_rate_list[t-1] = accept_rate
        K_list[t-1] = k
        # if len(A_t) > N:
        #     A_t = A_t[:N]
        particles_curr = np.array(A_t)
        # if np.isnan(particles_curr).any():
        #     print("发现问题了")
        #     print(particles_curr)

        x_est[t-1] = np.mean(particles_curr)
        # print(x_est)
        particles_prev = particles_curr.copy()

    runtime = time.time() - start
    return x_est, K_list, runtime, accept_rate_list

# ======================
# 6. APF 实现模块
# ======================
def APF(y_obs, N, Q, R, x0, prior_std=2.0):
    """
    辅助粒子滤波（Auxiliary Particle Filter）
    Args:
        y_obs:       观测序列 (长度 T)
        N:           粒子数
        Q, R:        过程噪声方差, 观测噪声方差
        x0:          初始状态均值
        prior_std:   初始粒子标准差
    Returns:
        x_est:       长度 T 的状态估计 (粒子均值)
        ESS:         长度 T 的有效样本数列表
        runtime:     总耗时 (秒)
    """
    T = len(y_obs)
    particles = np.random.normal(x0, prior_std, size=N)
    weights = np.ones(N) / N
    x_est = np.zeros(T)
    ESS = np.zeros(T)

    start = time.time()
    for t in range(1, T + 1):
        # 1) 计算预测均值，用于辅助变量引导
        pred_mean = f_xt(particles, t)

        # 2) 计算辅助权重：w̃ ∝ w_{t-1}^{(i)} * p(y_t | pred_mean)
        aux_lik = norm.pdf(y_obs[t - 1], loc=h_zt(pred_mean), scale=np.sqrt(R))
        aux_weights = weights * aux_lik
        aux_weights += 1e-300
        aux_weights /= np.sum(aux_weights)

        # 3) 从辅助分布中重采样 index
        indices = np.random.choice(N, size=N, replace=True, p=aux_weights)
        particles_resampled = particles[indices]

        # 4) 生成真实状态粒子
        particles = f_xt(particles_resampled, t) + np.random.normal(0, np.sqrt(Q), size=N)

        # 5) 更新权重：使用真实粒子的观测似然，但除以辅助似然进行修正
        new_lik = norm.pdf(y_obs[t - 1], loc=h_zt(particles), scale=np.sqrt(R)) + 1e-300
        # new_lik = compute_likelihood(particles, y_obs[t - 1], R)
        ref_lik = norm.pdf(y_obs[t - 1], loc=h_zt(f_xt(particles_resampled, t)), scale=np.sqrt(R)) + 1e-300
        weights = new_lik / ref_lik
        weights += 1e-300
        weights /= np.sum(weights)

        # 6) 估计 & ESS
        x_est[t - 1] = np.sum(particles * weights)
        ESS[t - 1] = 1.0 / np.sum(weights ** 2)

        # 可选的再次重采样（如 ESS 太小）
        if ESS[t - 1] < N / 1.0:
            indices = np.random.choice(N, size=N, replace=True, p=weights)
            particles = particles[indices]
            weights.fill(1.0 / N)

    runtime = time.time() - start
    return x_est, ESS, runtime




# ======================
# 7. 主程序：100 次蒙特卡洛仿真
# ======================
if __name__ == "__main__":
    # —— 超参数 ——
    T = 50                 # 仿真时长
    Q_true = 0.1           # 过程噪声方差
    R_true = 0.1           # 观测噪声方差
    x0 = 5.0               # 初始状态
    N = 100           # 粒子数
    prior_std = 2.0        # 初始粒子方差

    alpha = 0.9            # BCPS‑PF 接受阈值
    max_batches = 50      # BCPS‑PF 最大批次数
    mc_runs = 10        # Monte Carlo 次数

    # 用于记录各算法在 mc_runs 次实验中的累计 RMSE、时间和 ESS
    rmse_sir, rmse_epf, rmse_upf, rmse_bcps, rmse_apf = [], [], [], [],[]
    # rmse_sir, rmse_bcps = [], []
    time_sir, time_bcps, time_epf, time_upf, time_apf = [], [], [], [],[]
    K_list_bcps = []
    ESS_sir_all = np.zeros((mc_runs, T))
    ESS_epf_all = np.zeros((mc_runs, T))
    ESS_upf_all = np.zeros((mc_runs, T))
    ESS_apf_all = np.zeros((mc_runs, T))

    errors_bcps = np.zeros((mc_runs, T))
    errors_sir = np.zeros((mc_runs, T))
    errors_epf = np.zeros((mc_runs, T))
    errors_upf = np.zeros((mc_runs, T))
    errors_apf = np.zeros((mc_runs, T))



    particle_counts = [10, 30, 50, 100,300,500]
    mc_runs_for_N_sweep = 2  # 对于每种粒子数量进行 500 次 Monte Carlo

    # 存储不同 N 值下的平均 RMSE
    avg_rmse_sir_N = []
    avg_rmse_epf_N = []
    avg_rmse_upf_N = []
    avg_rmse_bcps_N = []
    avg_rmse_apf_N = []

    # 存储 N=100 时的每个时间步的 RMSE 曲线
    rmse_sir_curve_N100 = None
    rmse_epf_curve_N100 = None
    rmse_upf_curve_N100 = None
    rmse_bcps_curve_N100 = None
    rmse_apf_curve_N100 = None
    avg_time_sir_N = []
    avg_time_epf_N = []
    avg_time_upf_N = []
    avg_time_bcps_N = []
    avg_time_apf_N = []

    avg_accept_rate_list_bcps = np.zeros((len(particle_counts), T))


    ff = 0
    print("开始不同粒子数量下的 Monte Carlo 仿真...")
    for current_N in particle_counts:
        print(f"\n===== 当前粒子数 N = {current_N}, 进行 {mc_runs_for_N_sweep} 轮 Monte Carlo 仿真 =====")

        rmse_sir, rmse_epf, rmse_upf, rmse_bcps, rmse_apf = [], [], [], [], []
        time_sir, time_bcps, time_epf, time_upf, time_apf = [], [], [], [], []
        K_list_bcps = []
        ESS_sir_all = np.zeros((mc_runs_for_N_sweep, T))
        ESS_epf_all = np.zeros((mc_runs_for_N_sweep, T))
        ESS_upf_all = np.zeros((mc_runs_for_N_sweep, T))
        ESS_apf_all = np.zeros((mc_runs_for_N_sweep, T))

        errors_bcps = np.zeros((mc_runs_for_N_sweep, T))
        errors_sir = np.zeros((mc_runs_for_N_sweep, T))
        errors_epf = np.zeros((mc_runs_for_N_sweep, T))
        errors_upf = np.zeros((mc_runs_for_N_sweep, T))
        errors_apf = np.zeros((mc_runs_for_N_sweep, T))

        accept_rate_list_bcps = np.zeros((mc_runs_for_N_sweep, T))


        # Monte Carlo 循环
        for run in range(mc_runs_for_N_sweep):
            if (run + 1) % 50 == 0:  # 每50轮打印一次，避免输出过多
                print(f'  N={current_N}: 第 {run + 1}/{mc_runs_for_N_sweep} 轮')
            seed = run + 1
            # 1) 数据模拟
            x_true, y_obs = simulate_kitagawa(T, Q_true, R_true, x0, seed=seed)

            # 2) SIR-PF
            x_sir, ESS_sir, t_sir = SIR_PF(y_obs, current_N, Q_true, R_true, x0, prior_std)
            errors_sir[run, :] = (x_sir - x_true) ** 2
            rmse_sir.append(np.sqrt(np.mean((x_true - x_sir) ** 2)))
            time_sir.append(t_sir)
            ESS_sir_all[run, :] = ESS_sir

            # 3) EKF-PF
            x_epf, ESS_epf, t_epf = EKF_PF(y_obs, current_N, Q_true, R_true, x0, P0=1.0)
            errors_epf[run, :] = (x_epf - x_true) ** 2
            rmse_epf.append(np.sqrt(np.mean((x_true - x_epf) ** 2)))
            time_epf.append(t_epf)
            ESS_epf_all[run, :] = ESS_epf

            # 4) UPF
            x_upf, ESS_upf, t_upf = UPF(y_obs, current_N, Q_true, R_true, x0, P0=1.0)
            errors_upf[run, :] = (x_upf - x_true) ** 2
            rmse_upf.append(np.sqrt(np.mean((x_true - x_upf) ** 2)))
            time_upf.append(t_upf)
            ESS_upf_all[run, :] = ESS_upf

            # 5) BCPS‑PF
            x_bcps, K_list, t_bcps,accept_rate = BCPS_PF(y_obs, current_N, Q_true, R_true,
                                             x0, alpha=alpha, prior_std=prior_std,
                                             max_batches=max_batches)
            K_list_bcps.append(K_list)
            errors_bcps[run, :] = (x_bcps - x_true) ** 2
            accept_rate_list_bcps[run,:] = accept_rate
            rmse_bcps.append(np.sqrt(np.mean((x_true - x_bcps) ** 2)))
            time_bcps.append(t_bcps)

            # 6) APF
            x_apf, ESS_apf, t_apf = APF(y_obs, current_N, Q_true, R_true, x0, prior_std)
            errors_apf[run, :] = (x_apf - x_true) ** 2
            rmse_apf.append(np.sqrt(np.mean((x_true - x_apf) ** 2)))
            time_apf.append(t_apf)
            ESS_apf_all[run, :] = ESS_apf

        # 记录当前 N 值下的平均 RMSE
        avg_rmse_sir_N.append(np.mean(rmse_sir))
        avg_rmse_epf_N.append(np.mean(rmse_epf))
        avg_rmse_upf_N.append(np.mean(rmse_upf))
        avg_rmse_bcps_N.append(np.mean(rmse_bcps))
        avg_rmse_apf_N.append(np.mean(rmse_apf))

        # 记录当前 N 值下的平均时间
        avg_time_sir_N.append(np.mean(time_sir))
        avg_time_epf_N.append(np.mean(time_epf))
        avg_time_upf_N.append(np.mean(time_upf))
        avg_time_bcps_N.append(np.mean(time_bcps))
        avg_time_apf_N.append(np.mean(time_apf))

        avg_accept_rate_list_bcps[ff,:] = np.mean(accept_rate_list_bcps, axis=0)
        ff += 1


        # 如果当前 N 为 100，则保存每个时间步的 RMSE 曲线
        if current_N == 100:
            rmse_sir_curve_N100 = np.sqrt(np.mean(errors_sir, axis=0))
            rmse_epf_curve_N100 = np.sqrt(np.mean(errors_epf, axis=0))
            rmse_upf_curve_N100 = np.sqrt(np.mean(errors_upf, axis=0))
            rmse_bcps_curve_N100 = np.sqrt(np.mean(errors_bcps, axis=0))
            rmse_apf_curve_N100 = np.sqrt(np.mean(errors_apf, axis=0))

    #保存不同粒子数条件下不同算法的平均时间，保存在一个excel表格里边，并设计表头
    import pandas as pd
    df_time = pd.DataFrame({
        'Particle Count': particle_counts,
        'SIR-PF Time (s)': avg_time_sir_N,
        'EPF Time (s)': avg_time_epf_N,
        'UPF Time (s)': avg_time_upf_N,
        'BCPS-PF Time (s)': avg_time_bcps_N,
        'APF Time (s)': avg_time_apf_N
    })
    df_time.to_excel('particle_count_time_comparison.xlsx', index=False)

    # 保存不同粒子数条件下不同算法的平均 RMSE，保存在一个excel表格里边，并设计表头
    df_rmse = pd.DataFrame({
        'Particle Count': particle_counts,
        'SIR-PF RMSE': avg_rmse_sir_N,
        'EPF RMSE': avg_rmse_epf_N,
        'UPF RMSE': avg_rmse_upf_N,
        'BCPS-PF RMSE': avg_rmse_bcps_N,
        'APF RMSE': avg_rmse_apf_N
    })
    df_rmse.to_excel('particle_count_rmse_comparison.xlsx', index=False)

    #保存当粒子数量 N=100 时，每个时间步的 RMSE，保存在一个excel表格里边，并设计表头
    df_rmse_N100 = pd.DataFrame({
        'Time Step': np.arange(1, T + 1),
        'SIR-PF RMSE': rmse_sir_curve_N100,
        'EPF RMSE': rmse_epf_curve_N100,
        'UPF RMSE': rmse_upf_curve_N100,
        'BCPS-PF RMSE': rmse_bcps_curve_N100,
        'APF RMSE': rmse_apf_curve_N100
    })
    df_rmse_N100.to_excel('rmse_per_step_N100.xlsx', index=False)

    # 保存 BCPS-PF 的 接受率，保存在一个excel表格里边，并设计表头
    df_accept_rate = pd.DataFrame({
        'Time Step': np.arange(1, T + 1),
        'Particle Count 10': avg_accept_rate_list_bcps[0],
        'Particle Count 30': avg_accept_rate_list_bcps[1],
        'Particle Count 50': avg_accept_rate_list_bcps[2],
        'Particle Count 100': avg_accept_rate_list_bcps[3],
        'Particle Count 300': avg_accept_rate_list_bcps[4],
        'Particle Count 500': avg_accept_rate_list_bcps[5]
    })
    df_accept_rate.to_excel('bcps_accept_rate_comparison.xlsx', index=False)





    # --- 绘图：当粒子数量 N=100 时，每个时间步的 RMSE 曲线 ---
    if rmse_sir_curve_N100 is not None:
        print("\n===== 绘制 N=100 时每个时间步的 RMSE 曲线 =====")
        t = np.arange(1, T + 1)

        plt.figure(figsize=(7, 4), dpi=300)
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4
        })

        plt.plot(t, rmse_sir_curve_N100, label='SIR-PF', color='orangered', linewidth=1.0, marker='s', markersize=3)
        plt.plot(t, rmse_apf_curve_N100, label='APF', color='darkorange', linewidth=1.0, marker='x', markersize=3)
        plt.plot(t, rmse_epf_curve_N100, label='EKF-PF', color='forestgreen', linewidth=1.0, marker='^', markersize=3)
        plt.plot(t, rmse_upf_curve_N100, label='UPF', color='darkcyan', linewidth=1.0, marker='d', markersize=3)
        plt.plot(t, rmse_bcps_curve_N100, label='BCPS-PF', color='royalblue', linewidth=1.0, marker='o', markersize=3)

        plt.xlabel('Time Step $t$', fontsize=13)
        plt.ylabel('RMSE', fontsize=13)
        plt.legend(frameon=True, loc='upper left', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        # plt.title('Per-step RMSE (N=100, 500 MC Runs)', fontsize=14)
        plt.tight_layout()
        plt.savefig('rmse_per_step_N100_plot.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    # --- 绘图：粒子数量 vs. RMSE 对比曲线 ---
    print("\n===== 绘制粒子数量 vs. RMSE 对比曲线 (横坐标为实际粒子数) =====")
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4
    })

    plt.plot(particle_counts, avg_rmse_sir_N, label='SIR-PF', color='orangered', linewidth=1.0, marker='s',
             markersize=4)
    plt.plot(particle_counts, avg_rmse_apf_N, label='APF', color='darkorange', linewidth=1.0, marker='x',
             markersize=4)
    plt.plot(particle_counts, avg_rmse_epf_N, label='EKF-PF', color='forestgreen', linewidth=1.0, marker='^',
             markersize=4)
    plt.plot(particle_counts, avg_rmse_upf_N, label='UPF', color='darkcyan', linewidth=1.0, marker='d',
             markersize=4)
    plt.plot(particle_counts, avg_rmse_bcps_N, label='BCPS-PF', color='royalblue', linewidth=1.0, marker='o',
             markersize=4)

    # 确保横坐标刻度只显示指定粒子数
    plt.xticks(particle_counts)
    plt.xlabel('Number of Particles (N)', fontsize=13)
    plt.ylabel('Average RMSE', fontsize=13)
    plt.legend(frameon=True, loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.title('Average RMSE vs. Number of Particles (500 MC Runs)', fontsize=14)
    plt.tight_layout()
    plt.savefig('rmse_vs_N_plot.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    #--- 绘图：粒子数量 vs. 平均时间 对比曲线 ---
    print("\n===== 绘制粒子数量 vs. 平均时间 对比曲线 (横坐标为实际粒子数) =====")
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4
    })

    plt.plot(particle_counts, avg_time_sir_N, label='SIR-PF', color='orangered', linewidth=1.0, marker='s',
             markersize=4)
    plt.plot(particle_counts, avg_time_apf_N, label='APF', color='darkorange', linewidth=1.0, marker='x',
             markersize=4)
    plt.plot(particle_counts, avg_time_epf_N, label='EKF-PF', color='forestgreen', linewidth=1.0, marker='^',
             markersize=4)
    plt.plot(particle_counts, avg_time_upf_N, label='UPF', color='darkcyan', linewidth=1.0, marker='d',
             markersize=4)
    plt.plot(particle_counts, avg_time_bcps_N, label='BCPS-PF', color='royalblue', linewidth=1.0, marker='o',
             markersize=4)

    # 确保横坐标刻度只显示指定粒子数
    plt.xticks(particle_counts)
    plt.xlabel('Number of Particles (N)', fontsize=13)
    plt.ylabel('Average Time (s)', fontsize=13)
    plt.legend(frameon=True, loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.title('Average Time vs. Number of Particles (500 MC Runs)', fontsize=14)
    plt.tight_layout()
    plt.savefig('time_vs_N_plot.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # --- 绘图：BCPS-PF 的接受率曲线 ---
    print("\n===== 绘制 BCPS-PF 的接受率曲线 (不同粒子数下) =====")
    t = np.arange(1, T + 1)
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4
    })

    plt.plot(t, avg_accept_rate_list_bcps[0,:], label='N=10', color='orangered', linewidth=1.0, marker='s', markersize=3)
    plt.plot(t, avg_accept_rate_list_bcps[1,:], label='N=20', color='darkorange', linewidth=1.0, marker='x', markersize=3)
    plt.plot(t, avg_accept_rate_list_bcps[2,:], label='N=50', color='forestgreen', linewidth=1.0, marker='^', markersize=3)
    plt.plot(t, avg_accept_rate_list_bcps[3,:], label='N=100', color='darkcyan', linewidth=1.0, marker='d', markersize=3)
    plt.plot(t, avg_accept_rate_list_bcps[4,:], label='N=300', color='royalblue', linewidth=1.0, marker='o', markersize=3)
    plt.plot(t, avg_accept_rate_list_bcps[5,:], label='N=500', color='purple', linewidth=1.0, marker='*', markersize=3)
    plt.xlabel('Time Step $t$', fontsize=13)
    plt.ylabel('Acceptance Rate', fontsize=13)
    plt.legend(frameon=True, loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.title('BCPS-PF Acceptance Rate vs. Time Step (500 MC Runs)', fontsize=14)
    plt.tight_layout()
    plt.savefig('bcps_acceptance_rate_plot.pdf', format='pdf', bbox_inches='tight')
    plt.show()


































