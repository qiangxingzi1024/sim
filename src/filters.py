# src/filters.py
import time

import numpy as np
from numba import njit, prange
from scipy.stats import norm

from src.models import f_xt, h_zt


# Helper function for likelihood calculation, used by multiple PFs
@njit(parallel=True)
def compute_likelihood(particles, y_obs_t, R):
    """
    Computes the likelihood (p(z_t | x_t^i)) for each particle.
    For the Kitagawa model, z_t = x_t^2 / 20 + noise.

    Args:
        particles (numpy.ndarray): Array of particles (x_t^i).
        y_obs_t (float): Current observation z_t.
        R (float): Observation noise variance.

    Returns:
        numpy.ndarray: Array of likelihoods for each particle.
    """
    n = particles.shape[0]
    lik = np.empty(n)
    sqrt_R = np.sqrt(R)
    for i in prange(n):
        h_val = h_zt(particles[i])  # 如果 h(x)=x，否则替换这里
        lik[i] = (1.0 / (np.sqrt(2 * np.pi) * sqrt_R)) * np.exp(-0.5 * ((y_obs_t - h_val) / sqrt_R) ** 2)
    return lik


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
    for t in range(1, T + 1):
        # 1) 预测：从 p(x_t | x_{t-1}) 采样
        particles = f_xt(particles, t) + np.random.normal(0, np.sqrt(Q), size=N)

        # 2) 更新权重：w ∝ p(y_t | x_t^{(i)})
        # lik = norm.pdf(y_obs[t-1], loc=h_zt(particles), scale=np.sqrt(R))
        lik = compute_likelihood(particles, y_obs[t - 1], R)
        weights = weights * lik
        weights += 1e-300  # 防止全为零
        weights /= np.sum(weights)

        # 3) 计算 ESS
        ESS[t - 1] = 1.0 / np.sum(weights ** 2)

        # 4) 重采样 (SIR)：按当前权重重新均匀抽取 N 个粒子
        idx = np.random.choice(N, size=N, replace=True, p=weights)
        particles = particles[idx]
        weights.fill(1.0 / N)

        # 5) 估计：取粒子均值
        x_est[t - 1] = np.mean(particles)
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
        return 0.5 + 25 * (1 - x ** 2) / (1 + x ** 2) ** 2

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
    for t in range(1, T + 1):
        # 每个粒子生成一个 proposal（EKF 更新）
        proposals = np.zeros(N)
        proposal_vars = np.zeros(N)
        for i in range(N):
            x_prev = particles[i]
            P_prev = P_particles[i]

            # 1) 预测步 (EKF)
            A = df_dx(x_prev, t)  # 雅可比
            x_pred = f_xt(x_prev, t)  # m_{t|t-1}
            P_pred = A * P_prev * A + Q  # P_{t|t-1}

            # 2) 更新步 (EKF)
            H = dh_dx(x_pred)
            if abs(H) < 1e-6:  # 防止过小
                H = 1e-6
            Kk = P_pred * H / (H * P_pred * H + R)  # 卡尔曼增益
            x_upd = x_pred + Kk * (y_obs[t - 1] - h_zt(x_pred))
            P_upd = (1 - Kk * H) * P_pred

            # 将 EKF 更新结果做 Proposal 分布 N(x_upd, P_upd)
            proposals[i] = np.random.normal(x_upd, np.sqrt(max(P_upd, 1e-8)))
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
            log_lik = norm.logpdf(y_obs[t - 1], loc=h_zt(proposals[i]), scale=np.sqrt(R))
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
        ESS[t - 1] = 1.0 / np.sum(weights ** 2)

        # 5) 重采样
        idx = np.random.choice(N, size=N, replace=True, p=weights)
        particles = proposals[idx]
        P_particles = P_particles[idx]
        weights.fill(1.0 / N)

        # 6) 估计
        x_est[t - 1] = np.mean(particles)
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
        cov_sqrt = np.sqrt((n + lam) * Pm)
        return np.array([xm, xm + cov_sqrt, xm - cov_sqrt])

    def compute_weights():
        # 单维的 2n+1 权重
        Wm = np.array([lam / (n + lam), 1 / (2 * (n + lam)), 1 / (2 * (n + lam))])
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
    for t in range(1, T + 1):
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
            P_pred = np.dot(Wc, (sigmas_pred - x_pred) ** 2) + Q

            # —— Unscented 更新 ——
            # 1) 构造预测 sigma
            sigmas2 = compute_sigma_points(x_pred, P_pred)
            # 2) 通过 h_zt 计算观测 sigma
            sigmas_obs = h_zt(sigmas2)
            # 3) 观测均值与协方差
            y_pred = np.dot(Wm, sigmas_obs)
            P_yy = np.dot(Wc, (sigmas_obs - y_pred) ** 2) + R
            P_xy = np.dot(Wc, (sigmas2 - x_pred) * (sigmas_obs - y_pred))

            # 4) 卡尔曼增益、更新均值与方差
            Kk = P_xy / P_yy
            x_upd = x_pred + Kk * (y_obs[t - 1] - y_pred)
            P_upd = P_pred - Kk * P_yy * Kk

            # 5) 将 UKF 更新结果当作 Proposal：N(x_upd, P_upd)
            proposals[i] = np.random.normal(x_upd, np.sqrt(max(P_upd, 1e-8)))
            proposal_vars[i] = P_upd

            # 缓存 P_upd
            P_particles[i] = P_upd

        # 6) 计算重要性权重：同 EKF-PF 部分
        log_w = np.zeros(N)
        for i in range(N):
            log_lik = norm.logpdf(y_obs[t - 1], loc=h_zt(proposals[i]), scale=np.sqrt(R))
            log_prior_trans = norm.logpdf(proposals[i], loc=f_xt(particles[i], t), scale=np.sqrt(Q))
            # Proposal 近似 N(x_upd, P_upd)
            log_q = norm.logpdf(proposals[i], loc=proposals[i], scale=np.sqrt(proposal_vars[i]))
            log_w[i] = log_lik + log_prior_trans - log_q

        c = np.max(log_w)
        weights = np.exp(log_w - c)
        weights += 1e-300
        weights /= np.sum(weights)

        # 7) 计算 ESS
        ESS[t - 1] = 1.0 / np.sum(weights ** 2)

        # 8) 重采样
        idx = np.random.choice(N, size=N, replace=True, p=weights)
        particles = proposals[idx]
        P_particles = P_particles[idx]
        weights.fill(1.0 / N)

        # 9) 状态估计
        x_est[t - 1] = np.mean(particles)

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
    Nt = N  # 初始粒子数
    accept_rate_list = np.zeros(T)

    start = time.time()
    for t in range(1, T + 1):
        A_t = []

        # 无噪声预测值
        base_particles = f_xt(particles_prev, t)

        # 不断迭代“批次”直到接受粒子 ≥ alpha*N 或 批次数到达 max_batches
        for k in range(1, max_batches + 1):
            # 1) 生成 N 个候选 = base + 过程噪声
            noise = np.random.normal(0, np.sqrt(Q), size=Nt)
            candidates = base_particles + noise

            # 2) 计算当前批次 N 个候选的观测似然
            lik = compute_likelihood(candidates, y_obs[t - 1], R)
            Lmax = 0.3989422804014327  # 1/sqrt(2*pi) 观测噪声的标准正态分布值

            # 3) 接受概率 = lik/Lmax
            r = np.random.rand(Nt)
            accepted = candidates[r < (lik / Lmax)]  # 接受的粒子
            # 补充一段代码，当 accepted 为空时，将lik中最大值和次大值对应的粒子添加到 accepted 中
            if len(accepted) == 0:
                max_lik_idx = np.argmax(lik)
                second_max_lik_idx = np.argsort(lik)[-2]
                accepted = np.array([candidates[max_lik_idx], candidates[second_max_lik_idx]])
            A_t.extend(accepted.tolist())

            if len(A_t) >= alpha * N:
                accept_rate = len(A_t) / (Nt * k)
                Nt = len(A_t)  # 更新当前批次粒子数
                break
            if k == max_batches:
                accept_rate = len(A_t) / (Nt * k)
                Nt = len(A_t)  # 达到最大批次数，停止采样
        accept_rate_list[t - 1] = accept_rate
        K_list[t - 1] = k
        particles_curr = np.array(A_t)

        x_est[t - 1] = np.mean(particles_curr)
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