#!/usr/bin/env python3
"""
5G-Advanced Joint 2D DRL Agent - Gymnasium (Gym) Socket IPC Co-Simulation Environment
File: ns3_environment.py

Architecture:
  - Real-time Socket IPC Server (127.0.0.1:5555)
  - Continuous Stateful ns-3 NR Physical Layer Simulation (No Restart per Step)
  - State Space (Box 10D): fixed-size load/backlog/channel statistics and control state
  - Action Space (Discrete 10D): 2D Joint Actions
    (5 x Delta_W x 2 x schedulable-PRB profile on one physical 100 MHz BWP)
"""

import os
import signal
import sys
import time
import socket
import subprocess
import numpy as np

OBSERVATION_DIM = 10
OBSERVATION_FIELDS = (
    "total_backlog",
    "max_ue_backlog",
    "offered_rate",
    "p99_latency",
    "mean_cqi",
    "p10_cqi",
    "min_cqi",
    "mean_dl_mcs",
    "active_window",
    # Legacy IPC field name: this selects the 20/100 MHz-equivalent PRB mask;
    # it is not a physical 5G-LENA BWP switch.
    "bwp_state",
)

try:
    import gymnasium as gym
    from gymnasium import spaces
    ENV_BASE_CLASS = gym.Env
except ImportError:
    try:
        import gym
        from gym import spaces
        ENV_BASE_CLASS = gym.Env
    except ImportError:
        class MockSpace: pass
        class Discrete(MockSpace):
            def __init__(self, n): self.n = n
            def sample(self): return int(np.random.randint(0, self.n))
        class Box(MockSpace):
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.array(low, dtype=dtype)
                self.high = np.array(high, dtype=dtype)
                self.shape = shape; self.dtype = dtype
            def sample(self): return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)
        class SpacesModule: Discrete = Discrete; Box = Box
        spaces = SpacesModule()
        class GymEnvBase:
            def __init__(self): pass
            def reset(self, seed=None, options=None): pass
            def step(self, action): pass
        ENV_BASE_CLASS = GymEnvBase


class Ns3MultiUserEnv(ENV_BASE_CLASS):
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 ns3_path="./ns3", 
                 num_ues=10, 
                 udp_rate=2.0, 
                 enable_burst_traffic=False,
                 synchronize_burst_traffic=False,
                 udp_burst_rate=3.0,
                 burst_mean_on_ms=80.0,
                 burst_mean_off_ms=120.0,
                 gnb_tx_power_dbm=30.0,
                 gnb_energy_bs_category="BsCat1",
                 gnb_energy_ref_config="Set1",
                 gnb_energy_eta_mode="Dual",
                 gnb_energy_antenna_ratio=0.4,
                 gnb_energy_power_unit_w=1.0,
                 ue_energy_power_unit_mw=1.0,
                 sim_time=20.0,
                 enable_mock=False,
                 port=5555,
                 csv_filename="multi_user_rl_step.csv",
                 qos_target_ms=30.0,
                 controller_warmup_ms=200.0,
                 controller_max_active_ms=30.0,
                 controller_initial_active_ms=30.0,
                 rng_seed=42,
                 weight_gnb_power=0.15,
                 weight_ue_power=0.15,
                 weight_latency=0.45,
                 weight_backlog=0.25):
        super(Ns3MultiUserEnv, self).__init__()

        self.ns3_path = ns3_path
        self.num_ues = num_ues
        self.udp_rate = udp_rate
        self.enable_burst_traffic = enable_burst_traffic
        self.synchronize_burst_traffic = synchronize_burst_traffic
        self.udp_burst_rate = udp_burst_rate
        self.burst_mean_on_ms = burst_mean_on_ms
        self.burst_mean_off_ms = burst_mean_off_ms
        self.gnb_tx_power_dbm = gnb_tx_power_dbm
        self.gnb_energy_bs_category = gnb_energy_bs_category
        self.gnb_energy_ref_config = gnb_energy_ref_config
        self.gnb_energy_eta_mode = gnb_energy_eta_mode
        self.gnb_energy_antenna_ratio = gnb_energy_antenna_ratio
        self.gnb_energy_power_unit_w = gnb_energy_power_unit_w
        self.ue_energy_power_unit_mw = ue_energy_power_unit_mw
        self.sim_time = sim_time
        self.enable_mock = enable_mock
        self.port = port
        self.csv_filename = csv_filename
        self.controller_warmup_ms = controller_warmup_ms
        self.controller_initial_active_ms = controller_initial_active_ms
        self.rng_seed = rng_seed
        self.episode_index = 0

        self.B_MAX_BYTES = 1000000.0
        self.B_MAX_PER_UE_BYTES = 200000.0
        self.OFFERED_RATE_MAX_MBPS = num_ues * (
            udp_rate + (udp_burst_rate if enable_burst_traffic else 0.0)
        )
        self.D_LIMIT_MS = 500.0
        self.CQI_MAX = 15.0
        self.DL_MCS_MAX = 28.0
        self.W_MIN_MS = 10.0
        self.W_MAX_MS = controller_max_active_ms
        official_gnb_power_sets = {
            "BsCat1": {"Set1": (55.0, 280.0), "Set2": (50.0, 200.0), "Set3": (38.0, 152.0)},
            "BsCat2": {"Set1": (5.5, 32.0), "Set2": (5.0, 26.0), "Set3": (3.0, 17.6)},
        }
        try:
            p3_relative, p4_relative = official_gnb_power_sets[
                gnb_energy_bs_category
            ][gnb_energy_ref_config]
        except KeyError as exc:
            raise ValueError(
                "Unsupported official gNB energy category/reference set: "
                f"{gnb_energy_bs_category}/{gnb_energy_ref_config}"
            ) from exc
        self.P_GNB_IDLE_W = p3_relative * gnb_energy_power_unit_w
        self.P_GNB_MAX_W = max(p4_relative * gnb_energy_power_unit_w, 1.0)
        self.P_UE_MAX_MW = max(300.0 * ue_energy_power_unit_mw, 1.0)
        self.QOS_SLA_TARGET_MS = qos_target_ms

        self.W_GNB_POWER = weight_gnb_power
        self.W_UE_POWER = weight_ue_power
        self.W_LATENCY = weight_latency
        self.W_BACKLOG = weight_backlog

        self.DELTA_W_OPTIONS = [-4.0, -2.0, 0.0, 2.0, 4.0]
        self.BWP_OPTIONS = [0, 1]
        self.action_space = spaces.Discrete(10)

        self.observation_space = spaces.Box(
            low=np.zeros(OBSERVATION_DIM, dtype=np.float32),
            high=np.ones(OBSERVATION_DIM, dtype=np.float32),
            shape=(OBSERVATION_DIM,),
            dtype=np.float32
        )

        self.current_w_active_ms = self.controller_initial_active_ms
        self.current_bwp_state = 0
        self.step_count = 0
        self.max_steps = 50

        self.server_socket = None
        self.client_conn = None
        self.client_stream = None
        self.ns3_process = None

    def _stop_ns3(self):
        """Close IPC resources and stop the complete ns-3 process group."""
        if self.client_stream is not None:
            try:
                self.client_stream.close()
            except OSError:
                pass
            self.client_stream = None

        if self.client_conn is not None:
            try:
                self.client_conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.client_conn.close()
            except OSError:
                pass
            self.client_conn = None

        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None

        process = self.ns3_process
        self.ns3_process = None
        if process is None or process.poll() is not None:
            return

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=3)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def _finish_ns3_episode(self):
        """Ask ns-3 to stop cleanly so it can export its final statistics."""
        process = self.ns3_process

        if self.client_stream is not None:
            try:
                self.client_stream.write(b"-1\n")
                self.client_stream.flush()
            except OSError:
                pass

        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._stop_ns3()
                raise RuntimeError("ns-3 did not exit cleanly after the terminal action")

        self._stop_ns3()

    def _start_socket_server_and_ns3(self):
        self._stop_ns3()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('127.0.0.1', self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(0.5)

        command = [
            self.ns3_path,
            "run",
            "scratch/joint-energy-mvp",
            "--",
            f"--simTime={self.sim_time}",
            f"--numUes={self.num_ues}",
            f"--udpDataRate={self.udp_rate}",
            f"--enableBurstTraffic={'true' if self.enable_burst_traffic else 'false'}",
            f"--synchronizeBurstTraffic={'true' if self.synchronize_burst_traffic else 'false'}",
            f"--udpBurstRate={self.udp_burst_rate}",
            f"--burstMeanOnMs={self.burst_mean_on_ms}",
            f"--burstMeanOffMs={self.burst_mean_off_ms}",
            f"--gnbTxPowerDbm={self.gnb_tx_power_dbm}",
            f"--gnbEnergyBsCategory={self.gnb_energy_bs_category}",
            f"--gnbEnergyRefConfig={self.gnb_energy_ref_config}",
            f"--gnbEnergyEtaMode={self.gnb_energy_eta_mode}",
            f"--gnbEnergyAntennaRatio={self.gnb_energy_antenna_ratio}",
            f"--gnbEnergyPowerUnitW={self.gnb_energy_power_unit_w}",
            f"--ueEnergyPowerUnitMw={self.ue_energy_power_unit_mw}",
            "--enableGating=true",
            "--enableQueueAware=false",
            "--enableCdrx=true",
            "--enableDciDynamic=true",
            "--enableBwpDynamic=true",
            "--enableRlIpc=true",
            f"--ipcPort={self.port}",
            f"--csvFileName={self.csv_filename}",
            f"--qosTargetDelayMs={self.QOS_SLA_TARGET_MS}",
            f"--controllerWarmupMs={self.controller_warmup_ms}",
            f"--controllerMaxActiveMs={self.W_MAX_MS}",
            f"--controllerInitialActiveMs={self.controller_initial_active_ms}",
            f"--cellDtxActiveMs={self.controller_initial_active_ms}",
            f"--rngSeed={self.rng_seed}",
            f"--rngRun={self.episode_index + 1}",
        ]

        self.ns3_process = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + 30.0

        try:
            while time.monotonic() < deadline:
                if self.ns3_process.poll() is not None:
                    exit_code = self.ns3_process.returncode
                    raise RuntimeError(
                        f"ns-3 exited before IPC connection (exit code {exit_code})"
                    )
                try:
                    self.client_conn, addr = self.server_socket.accept()
                    self.client_conn.settimeout(10.0)
                    self.client_stream = self.client_conn.makefile("rwb")
                    print(
                        f"✅ ns-3 IPC connected: backend=ns3-ipc, "
                        f"port={self.port}, pid={self.ns3_process.pid}, peer={addr}"
                    )
                    return
                except socket.timeout:
                    continue
            raise TimeoutError("ns-3 did not connect to the IPC server within 30 seconds")
        except Exception:
            self._stop_ns3()
            raise

    def _action_to_config(self, action_idx):
        delta_w = self.DELTA_W_OPTIONS[action_idx % 5]
        bwp_next = self.BWP_OPTIONS[action_idx // 5]
        return delta_w, bwp_next

    def _receive_ns3_state(self):
        if self.client_stream is None:
            raise RuntimeError("ns-3 IPC stream is not connected")

        data = self.client_stream.readline()
        if not data:
            raise ConnectionError("ns-3 closed the IPC connection")

        parts = data.decode("utf-8").strip().split(",")
        if len(parts) != 15 or parts[0] != "S4":
            raise ValueError(f"invalid ns-3 state message: {data!r}")

        return {
            "backlog_bytes": float(parts[1]),
            "max_ue_backlog_bytes": float(parts[2]),
            "offered_rate_mbps": float(parts[3]),
            "delivered_rate_mbps": float(parts[4]),
            "p99_latency_ms": float(parts[5]),
            "avg_cqi": float(parts[6]),
            "p10_cqi": float(parts[7]),
            "min_cqi": float(parts[8]),
            "avg_dl_mcs": float(parts[9]),
            "w_active_ms": float(parts[10]),
            "bwp_state": int(parts[11]),
            "gnb_power_w": float(parts[12]),
            "ue_power_mw": float(parts[13]),
            "jain_fairness": float(parts[14]),
        }

    def _get_normalized_obs(
        self,
        rlc_backlog_bytes,
        max_ue_backlog_bytes,
        offered_rate_mbps,
        max_pending_delay_ms,
        avg_cqi,
        p10_cqi,
        min_cqi,
        avg_dl_mcs,
        w_active_ms,
        bwp_state,
    ):
        s1 = np.clip(rlc_backlog_bytes / self.B_MAX_BYTES, 0.0, 1.0)
        s2 = np.clip(max_ue_backlog_bytes / self.B_MAX_PER_UE_BYTES, 0.0, 1.0)
        s3 = np.clip(offered_rate_mbps / self.OFFERED_RATE_MAX_MBPS, 0.0, 1.0)
        s4 = np.clip(max_pending_delay_ms / self.D_LIMIT_MS, 0.0, 1.0)
        s5 = np.clip(avg_cqi / self.CQI_MAX, 0.0, 1.0)
        s6 = np.clip(p10_cqi / self.CQI_MAX, 0.0, 1.0)
        s7 = np.clip(min_cqi / self.CQI_MAX, 0.0, 1.0)
        s8 = np.clip(avg_dl_mcs / self.DL_MCS_MAX, 0.0, 1.0)
        s9 = np.clip((w_active_ms - self.W_MIN_MS) / (self.W_MAX_MS - self.W_MIN_MS), 0.0, 1.0)
        s10 = float(bwp_state)
        return np.array([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10], dtype=np.float32)

    def _compute_reward(self, gnb_power_watts, ue_avg_power_mw, p99_latency_ms, backlog_bytes):
        p_gnb_norm = np.clip(gnb_power_watts / self.P_GNB_MAX_W, 0.0, 1.0)
        p_ue_norm = np.clip(ue_avg_power_mw / self.P_UE_MAX_MW, 0.0, 1.0)
        backlog_norm = np.clip(backlog_bytes / self.B_MAX_BYTES, 0.0, 1.0)

        latency_excess_ratio = max(
            0.0,
            (p99_latency_ms - self.QOS_SLA_TARGET_MS) / self.QOS_SLA_TARGET_MS,
        )
        # Smoothly bound the latency term to [0, 1). This keeps PPO returns on
        # the same scale as the normalized power and backlog objectives while
        # still increasing monotonically for every SLA violation.
        penalty_p99 = 1.0 - np.exp(-latency_excess_ratio)

        reward = - (self.W_GNB_POWER * p_gnb_norm +
                    self.W_UE_POWER * p_ue_norm +
                    self.W_LATENCY * penalty_p99 +
                    self.W_BACKLOG * backlog_norm)

        return float(reward), {
            "p_gnb_norm": p_gnb_norm,
            "p_ue_norm": p_ue_norm,
            "penalty_p99": penalty_p99,
            "latency_excess_ratio": latency_excess_ratio,
            "backlog_norm": backlog_norm,
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng_seed = int(seed)
        self.step_count = 0
        self.current_w_active_ms = self.controller_initial_active_ms
        self.current_bwp_state = 0

        if not self.enable_mock:
            if not os.path.exists(self.ns3_path):
                raise FileNotFoundError(f"ns-3 wrapper not found: {self.ns3_path}")
            self._start_socket_server_and_ns3()
            self.episode_index += 1

        if self.enable_mock:
            initial_backlog = 1500.0
            initial_max_ue_backlog = 250.0
            initial_offered_rate = self.num_ues * self.udp_rate
            initial_delay = 8.5
            initial_cqi = 12.0
            initial_p10_cqi = 10.0
            initial_min_cqi = 9.0
            initial_avg_mcs = 20.0
        else:
            try:
                initial_state = self._receive_ns3_state()
            except (OSError, ValueError) as exc:
                self._stop_ns3()
                raise RuntimeError("failed to receive initial ns-3 state") from exc

            initial_backlog = initial_state["backlog_bytes"]
            initial_max_ue_backlog = initial_state["max_ue_backlog_bytes"]
            initial_offered_rate = initial_state["offered_rate_mbps"]
            initial_delay = initial_state["p99_latency_ms"]
            initial_cqi = initial_state["avg_cqi"]
            initial_p10_cqi = initial_state["p10_cqi"]
            initial_min_cqi = initial_state["min_cqi"]
            initial_avg_mcs = initial_state["avg_dl_mcs"]
            self.current_w_active_ms = initial_state["w_active_ms"]
            self.current_bwp_state = initial_state["bwp_state"]

        obs = self._get_normalized_obs(
            rlc_backlog_bytes=initial_backlog,
            max_ue_backlog_bytes=initial_max_ue_backlog,
            offered_rate_mbps=initial_offered_rate,
            max_pending_delay_ms=initial_delay,
            avg_cqi=initial_cqi,
            p10_cqi=initial_p10_cqi,
            min_cqi=initial_min_cqi,
            avg_dl_mcs=initial_avg_mcs,
            w_active_ms=self.current_w_active_ms,
            bwp_state=self.current_bwp_state
        )

        info = {
            "w_active_ms": self.current_w_active_ms,
            "bwp_state": self.current_bwp_state,
            "backend": "mock" if self.enable_mock else "ns3-ipc",
            "ipc_port": None if self.enable_mock else self.port,
            "message": "Environment reset successfully."
        }

        return obs, info

    def step(self, action):
        self.step_count += 1
        delta_w, bwp_next = self._action_to_config(action)

        backlog_bytes = 2500.0
        max_ue_backlog_bytes = 400.0
        offered_rate_mbps = self.num_ues * self.udp_rate
        delivered_rate_mbps = 0.0
        p99_latency_ms = 22.0
        avg_cqi = 12.0
        p10_cqi = 10.0
        min_cqi = 9.0
        avg_dl_mcs = 20.0
        gnb_power_w = 24.5
        ue_power_mw = 55.0
        jain_fairness = 0.74

        if not self.enable_mock:
            try:
                self.client_stream.write(f"{action}\n".encode("utf-8"))
                self.client_stream.flush()

                state = self._receive_ns3_state()
                backlog_bytes = state["backlog_bytes"]
                max_ue_backlog_bytes = state["max_ue_backlog_bytes"]
                offered_rate_mbps = state["offered_rate_mbps"]
                delivered_rate_mbps = state["delivered_rate_mbps"]
                p99_latency_ms = state["p99_latency_ms"]
                avg_cqi = state["avg_cqi"]
                p10_cqi = state["p10_cqi"]
                min_cqi = state["min_cqi"]
                avg_dl_mcs = state["avg_dl_mcs"]
                self.current_w_active_ms = state["w_active_ms"]
                self.current_bwp_state = state["bwp_state"]
                gnb_power_w = state["gnb_power_w"]
                ue_power_mw = state["ue_power_mw"]
                jain_fairness = state["jain_fairness"]
            except (OSError, ValueError, IndexError) as exc:
                self._stop_ns3()
                raise RuntimeError(
                    f"ns-3 IPC failed at environment step {self.step_count}"
                ) from exc

        if self.enable_mock:
            self.current_w_active_ms = float(np.clip(
                self.current_w_active_ms + delta_w, self.W_MIN_MS, self.W_MAX_MS
            ))
            self.current_bwp_state = int(bwp_next)

            capacity_factor = (self.current_w_active_ms / 10.0) * (1.0 + 1.5 * self.current_bwp_state)
            backlog_bytes = max(0.0, 15000.0 - capacity_factor * 3500.0 + np.random.normal(0, 500))
            max_ue_backlog_bytes = backlog_bytes / max(1.0, self.num_ues * 0.7)
            if self.enable_burst_traffic:
                burst_fraction = np.random.beta(2.0, 3.0)
                offered_rate_mbps += (
                    self.num_ues * self.udp_burst_rate * burst_fraction
                )
            max_delay_ms = max(5.0, 80.0 / capacity_factor + np.random.normal(0, 2.0))
            delivered_rate_mbps = min(
                offered_rate_mbps,
                20.0 if self.current_bwp_state == 0 else 100.0,
            )
            avg_cqi = 12.0 + np.random.uniform(-0.5, 0.5)
            p10_cqi = max(0.0, avg_cqi - 2.0)
            min_cqi = max(0.0, avg_cqi - 3.0)
            avg_dl_mcs = np.clip(2.0 * avg_cqi - 4.0, 0.0, self.DL_MCS_MAX)

            activity = np.clip(self.current_w_active_ms / 30.0, 0.0, 1.0)
            bandwidth_scale = 0.2 if self.current_bwp_state == 0 else 1.0
            gnb_power_w = self.P_GNB_IDLE_W + (
                self.P_GNB_MAX_W - self.P_GNB_IDLE_W
            ) * activity * bandwidth_scale
            ue_power_mw = self.ue_energy_power_unit_mw * (
                45.0 + (300.0 - 45.0) * activity * bandwidth_scale
            )
            p99_latency_ms = max_delay_ms * 1.3
            jain_fairness = 0.74 + 0.02 * np.random.uniform(-1, 1)

        obs = self._get_normalized_obs(
            rlc_backlog_bytes=backlog_bytes,
            max_ue_backlog_bytes=max_ue_backlog_bytes,
            offered_rate_mbps=offered_rate_mbps,
            max_pending_delay_ms=p99_latency_ms,
            avg_cqi=avg_cqi,
            p10_cqi=p10_cqi,
            min_cqi=min_cqi,
            avg_dl_mcs=avg_dl_mcs,
            w_active_ms=self.current_w_active_ms,
            bwp_state=self.current_bwp_state
        )

        reward, reward_breakdown = self._compute_reward(
            gnb_power_watts=gnb_power_w,
            ue_avg_power_mw=ue_power_mw,
            p99_latency_ms=p99_latency_ms,
            backlog_bytes=backlog_bytes,
        )

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "action_idx": action,
            "delta_w_ms": delta_w,
            "w_active_ms": self.current_w_active_ms,
            "bwp_state": self.current_bwp_state,
            "backlog_bytes": backlog_bytes,
            "max_ue_backlog_bytes": max_ue_backlog_bytes,
            "offered_rate_mbps": offered_rate_mbps,
            "delivered_rate_mbps": delivered_rate_mbps,
            "p99_latency_ms": p99_latency_ms,
            "avg_cqi": avg_cqi,
            "p10_cqi": p10_cqi,
            "min_cqi": min_cqi,
            "avg_dl_mcs": avg_dl_mcs,
            "gnb_power_w": gnb_power_w,
            "ue_power_mw": ue_power_mw,
            "jain_fairness": jain_fairness,
            "backend": "mock" if self.enable_mock else "ns3-ipc",
            "ipc_port": None if self.enable_mock else self.port,
            "reward_breakdown": reward_breakdown
        }

        if truncated and not self.enable_mock:
            self._finish_ns3_episode()

        return obs, reward, terminated, truncated, info

    def close(self):
        self._stop_ns3()
