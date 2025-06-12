#phoenix_loop_robustness_sim.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d # For 3D plots
from collections import Counter
from scipy.stats import entropy as shannon_entropy # Renamed to avoid conflict
from phoenix_loop_scenarios_sim import DEFAULT_PARAMS_MULTI 
# --- Agent and ABM System ---
class SimpleAgent:
    def __init__(self, agent_id, num_activity_levels, initial_activity_state=None):
        self.agent_id = agent_id
        self.num_activity_levels = num_activity_levels
        if initial_activity_state is None:
            self.activity_state = np.random.randint(0, num_activity_levels)
        else:
            self.activity_state = initial_activity_state

    def step(self, exploration_propensity): # exploration_propensity linked to macro betaLever
        """
        exploration_propensity: float between 0 (deterministic) and 1 (fully random).
        Low macro betaLever (Flaring) -> high exploration_propensity.
        High macro betaLever (Stable) -> low exploration_propensity.
        """
        if np.random.rand() < exploration_propensity:
            # Explore: pick any state randomly
            self.activity_state = np.random.randint(0, self.num_activity_levels)
        else:
            # Exploit/Constrained: tend to stay or move to adjacent states
            # (Simple rule: 50% stay, 25% move to state-1, 25% move to state+1, with wrap-around)
            rand_choice = np.random.rand()
            if rand_choice < 0.5:
                pass # Stay
            elif rand_choice < 0.75:
                self.activity_state = (self.activity_state - 1 + self.num_activity_levels) % self.num_activity_levels
            else:
                self.activity_state = (self.activity_state + 1) % self.num_activity_levels
        return self.activity_state

class ABMSystem:
    def __init__(self, num_agents, num_activity_levels):
        self.num_agents = num_agents
        self.num_activity_levels = num_activity_levels
        self.agents = [SimpleAgent(i, num_activity_levels) for i in range(num_agents)]
        self.current_activity_states = np.array([agent.activity_state for agent in self.agents])

    def step_abm(self, exploration_propensity):
        self.current_activity_states = np.array([agent.step(exploration_propensity) for agent in self.agents])

    def get_entropy_exp_shannon(self):
        if len(self.current_activity_states) == 0: return 0.0
        counts = np.bincount(self.current_activity_states, minlength=self.num_activity_levels)
        probabilities = counts / len(self.current_activity_states)
        # Use scipy.stats.entropy for Shannon entropy
        return shannon_entropy(probabilities, base=2)


    def get_entropy_exp_variance(self):
        if len(self.current_activity_states) < 2: return 0.0 # Variance needs at least 2 points
        return np.var(self.current_activity_states)

    def get_entropy_exp_range(self):
        if len(self.current_activity_states) == 0: return 0.0
        return np.max(self.current_activity_states) - np.min(self.current_activity_states)

# --- PhoenixLoopSimulator (Modified to include ABM and entropy choices) ---
class PhoenixLoopSimulatorWithABM:
    def __init__(self, params, abm_system_instance=None): # Pass ABM instance
        self.params = params.copy() # Use a copy to allow modifications per run
        self.t = 0.0

        self.gLever = params['g0']
        self.betaLever = params['beta0']
        self.FEcrit = params['Fcrit0']
        
        # ABM setup
        self.abm_system = abm_system_instance
        if self.abm_system is None: # Default ABM if none provided
            self.abm_system = ABMSystem(
                params.get('num_abm_agents', 100),
                params.get('num_abm_activity_levels', 10)
            )
        
        self.entropy_exp_method = params.get('entropy_exp_method', 'shannon') # 'shannon', 'variance', 'range'
        self.EntropyBaseline = params.get('EntropyBaseline', 1.0) # This will be set based on baseline runs

        self.avg_delta_P_tau = self.params['strain_profile_func'](self.t, params, self)
        self.ThetaT = self._calculate_ThetaT()
        
        self.EntropyExp = self._calculate_EntropyExp_from_abm()
        self.rhoE = self._calculate_rhoE(self.EntropyExp)

        self.is_collapsed = False
        self.collapse_time = -1.0
        self.current_phase_est = "PRE_COLLAPSE"

        self.history = []
        self._record_history()

        self.fcrit_shock_applied = False 
        self.lever_cost_escalation_active = False 
        self.original_k_g = params.get('k_g', 0.1)

    def _map_macro_beta_to_abm_exploration(self):
        # Inverse relationship: low macro beta (Flaring) means high ABM exploration
        # High macro beta (Stable) means low ABM exploration
        # Let's try a simple linear mapping, needs tuning
        # Max exploration when betaLever is at beta_exploratory
        # Min exploration when betaLever is at beta_normal_op_target or beta_restabilization_target
        
        min_macro_beta = self.params.get('beta_exploratory', 0.1)
        max_macro_beta = max(self.params.get('beta_normal_op_target',2.0), 
                             self.params.get('beta_restabilization_target',1.8))
        
        # Normalize current betaLever to [0,1] where 0 is min_macro_beta and 1 is max_macro_beta
        if max_macro_beta - min_macro_beta > 1e-6:
            normalized_beta = (self.betaLever - min_macro_beta) / (max_macro_beta - min_macro_beta)
            normalized_beta = np.clip(normalized_beta, 0, 1)
        else: # Avoid division by zero if min and max are too close
            normalized_beta = 0.5 if self.betaLever <= min_macro_beta else 1.0

        # Invert to get exploration propensity (0=low exploration, 1=high exploration)
        exploration_propensity = 1.0 - normalized_beta
        return np.clip(exploration_propensity, 0.05, 0.95) # Clamp to ensure some base randomness/determinism


    def _calculate_EntropyExp_from_abm(self):
        if self.entropy_exp_method == 'shannon':
            return self.abm_system.get_entropy_exp_shannon()
        elif self.entropy_exp_method == 'variance':
            return self.abm_system.get_entropy_exp_variance()
        elif self.entropy_exp_method == 'range':
            return self.abm_system.get_entropy_exp_range()
        else: # Default to shannon
            return self.abm_system.get_entropy_exp_shannon()

    # ... (reuse _calculate_ThetaT, _calculate_lever_costs, _update_FEcrit, _calculate_rhoE, 
    #          _set_phase_targets, _update_phase_estimation from previous script) ...
    # The methods below are direct copies or very slightly adapted from your previous script.
    # Ensure these are consistent if you have made changes there.

    def _calculate_ThetaT(self):
        g = max(self.gLever, 1e-6); b = max(self.betaLever, 1e-6); f = max(self.FEcrit, 1e-6)
        return self.params['C'] * (g**self.params['w1']) * (b**self.params['w2']) * (f**self.params['w3'])

    def _calculate_lever_costs(self):
        current_k_g = self.params['k_g']
        if self.lever_cost_escalation_active: current_k_g = self.params.get('k_g_escalated', self.params['k_g']*10)
        cost_g = current_k_g * (self.gLever**self.params['phi1'])
        cost_beta = self.params['k_b'] * (self.betaLever**self.params['phi_b'])
        return cost_g, cost_beta

    def _update_FEcrit(self, cost_g, cost_beta, current_base_cost):
        dFEcrit_dt = self.params['F_influx_rate'] - current_base_cost - cost_g - cost_beta
        self.FEcrit += dFEcrit_dt * self.params['dt']; self.FEcrit = max(0, self.FEcrit)

    def _calculate_rhoE(self, current_EntropyExp):
        if self.EntropyBaseline > 1e-9: # Avoid division by very small number if baseline is near zero
            return current_EntropyExp / self.EntropyBaseline
        # If baseline is effectively zero, and current is also zero, rhoE is 1. If current is >0, rhoE is large.
        return 1.0 if abs(current_EntropyExp) < 1e-9 else current_EntropyExp / 1e-9


    def _set_phase_targets(self): # Identical to previous script
        target_g = self.gLever; target_beta = self.betaLever
        current_base_cost = self.params['F_base_cost_normal']
        if self.current_phase_est == "PRE_COLLAPSE":
            target_g = self.params['g_normal_op_target']; target_beta = self.params['beta_normal_op_target']
            if self.params.get('pre_collapse_drift_Fcrit_rate',0)!=0: self.FEcrit += self.params['pre_collapse_drift_Fcrit_rate']*self.params['dt']
            if self.params.get('pre_collapse_drift_beta_rate',0)!=0: self.betaLever += self.params['pre_collapse_drift_beta_rate']*self.params['dt']
        elif self.current_phase_est == "DISINTEGRATION": current_base_cost = self.params['F_base_cost_recovery'] 
        elif self.current_phase_est == "FLARING":
            target_beta = self.params['beta_exploratory']
            if self.FEcrit < self.params['Fcrit_g_reduction_threshold']: target_g = self.params['g_exploratory_low_fcrit']
            else: target_g = self.params['g_exploratory']
            current_base_cost = self.params['F_base_cost_recovery']
        elif self.current_phase_est == "PRUNING":
            target_beta = self.params['beta_pruning_target']; target_g = self.params['g_pruning_target']
            current_base_cost = self.params['F_base_cost_recovery'] 
        elif self.current_phase_est == "RESTABILIZATION":
            target_beta = self.params['beta_restabilization_target']; target_g = self.params['g_restabilization_target']
        return target_g, target_beta, current_base_cost

    def _update_phase_estimation(self): # Identical to previous script
        collapse_condition_met = False
        if self.FEcrit <= self.params['FEcrit_min_abs']: collapse_condition_met = True
        elif self.ThetaT > 1e-6 and (self.avg_delta_P_tau / self.ThetaT > self.params['strain_collapse_ratio']): collapse_condition_met = True
        elif self.params.get('beta_collapse_threshold', float('inf')) <= self.betaLever : collapse_condition_met = True
        if not self.is_collapsed and collapse_condition_met:
            self.is_collapsed = True; self.collapse_time = self.t; self.current_phase_est = "DISINTEGRATION"
        if self.is_collapsed:
            time_since_collapse_start = self.t - self.collapse_time
            dis_dur = self.params.get('disintegration_duration', self.params['dt'])
            if self.current_phase_est == "DISINTEGRATION":
                if time_since_collapse_start >= dis_dur: self.current_phase_est = "FLARING"
            elif self.current_phase_est == "FLARING":
                time_in_flaring = time_since_collapse_start - dis_dur
                if time_in_flaring >= self.params['min_flaring_duration'] and self.FEcrit >= self.params['Fcrit_pruning_threshold']: self.current_phase_est = "PRUNING"
            elif self.current_phase_est == "PRUNING":
                time_in_pruning = time_since_collapse_start - dis_dur - self.params['min_flaring_duration']
                beta_stab = abs(self.betaLever - self.params['beta_pruning_target']) < 0.20 * self.params['beta_pruning_target'] 
                if time_in_pruning >= self.params['min_pruning_duration'] and self.FEcrit >= self.params['Fcrit_restabilization_threshold'] and beta_stab : self.current_phase_est = "RESTABILIZATION"


    def step(self):
        # Scenario-specific triggers (Fcrit shock, lever cost escalation)
        t_fcrit_shock = self.params.get('t_fcrit_shock', -1)
        if t_fcrit_shock >= 0 and abs(self.t - t_fcrit_shock) < self.params['dt']/2 and not self.fcrit_shock_applied:
            self.FEcrit = max(0, self.FEcrit - self.params.get('fcrit_shock_abs_drop', 0))
            self.fcrit_shock_applied = True

        t_lever_cost_esc_start = self.params.get('t_lever_cost_esc_start', -1)
        t_lever_cost_esc_end = self.params.get('t_lever_cost_esc_end', -1)
        if t_lever_cost_esc_start >=0 and self.t >= t_lever_cost_esc_start and (t_lever_cost_esc_end < 0 or self.t < t_lever_cost_esc_end):
            if not self.lever_cost_escalation_active: self.lever_cost_escalation_active = True
        elif self.lever_cost_escalation_active and (t_lever_cost_esc_end >= 0 and self.t >= t_lever_cost_esc_end):
            self.lever_cost_escalation_active = False
            
        # Update macro system
        self.avg_delta_P_tau = self.params['strain_profile_func'](self.t, self.params, self)
        self._update_phase_estimation() 
        target_g, target_beta, current_base_cost = self._set_phase_targets()

        self.gLever += (target_g - self.gLever) * self.params['g_adaptation_rate'] * self.params['dt']
        self.betaLever += (target_beta - self.betaLever) * self.params['beta_adaptation_rate'] * self.params['dt']
        
        self.gLever = max(self.params['g_min'], min(self.params['g_max'], self.gLever))
        self.betaLever = max(self.params['beta_min'], min(self.params['beta_max'], self.betaLever))

        cost_g, cost_beta = self._calculate_lever_costs()
        self._update_FEcrit(cost_g, cost_beta, current_base_cost)
        self.ThetaT = self._calculate_ThetaT()
        
        # Update ABM based on current macro betaLever
        abm_exploration_prop = self._map_macro_beta_to_abm_exploration()
        self.abm_system.step_abm(abm_exploration_prop)
        
        # Calculate EntropyExp from ABM and then rhoE
        self.EntropyExp = self._calculate_EntropyExp_from_abm()
        self.rhoE = self._calculate_rhoE(self.EntropyExp)
        
        self._record_history()
        self.t += self.params['dt']

    def _record_history(self): # Identical to previous script
        phase_numeric_map = {"PRE_COLLAPSE":0, "DISINTEGRATION":1, "FLARING":2, "PRUNING":3, "RESTABILIZATION":4}
        self.history.append({
            't': self.t, 'gLever': self.gLever, 'betaLever': self.betaLever, 'FEcrit': self.FEcrit,
            'avg_delta_P_tau': self.avg_delta_P_tau, 'ThetaT': self.ThetaT,
            'is_collapsed': int(self.is_collapsed), 
            'current_phase_est_numeric': phase_numeric_map.get(self.current_phase_est, -1), 
            'rhoE': self.rhoE, 'EntropyExp': self.EntropyExp,
        })

    def run_simulation(self): # Largely identical, ensure robustness for short DFs
        self.fcrit_shock_applied = False
        self.lever_cost_escalation_active = False
        
        while self.t < self.params['T_max']:
            if self.is_collapsed and self.current_phase_est == "RESTABILIZATION":
                time_in_restabilization = self.t - (self.collapse_time + self.params.get('disintegration_duration',0) + self.params['min_flaring_duration'] + self.params['min_pruning_duration'])
                if time_in_restabilization > self.params.get("stabilization_duration_check", 50): break 
            self.step()
        
        df = pd.DataFrame(self.history)
        if df.empty or len(df) < 2: return df

        df['dot_gLever'] = df['gLever'].diff()/self.params['dt']; df['dot_betaLever'] = df['betaLever'].diff()/self.params['dt']
        df['dot_FEcrit'] = df['FEcrit'].diff()/self.params['dt']
        for col in ['dot_gLever','dot_betaLever','dot_FEcrit']: df[col] = df[col].fillna(0)

        smooth_window = self.params.get('W_smooth_deriv',1)
        if smooth_window > 1 and len(df) >= smooth_window :
            df['dot_betaLever_smooth'] = df['dot_betaLever'].rolling(window=smooth_window, center=True, min_periods=1).mean()
            df['dot_FEcrit_smooth'] = df['dot_FEcrit'].rolling(window=smooth_window, center=True, min_periods=1).mean()
        else:
            df['dot_betaLever_smooth'] = df['dot_betaLever']; df['dot_FEcrit_smooth'] = df['dot_FEcrit']
            
        df['SpeedIndex'] = np.sqrt(df['dot_betaLever_smooth']**2 + df['dot_FEcrit_smooth']**2)
        W_couple = self.params.get('W_couple',20); min_periods_couple = max(2, W_couple//2) 
        if len(df) >= min_periods_couple:
            df['CoupleIndex'] = df['dot_betaLever_smooth'].rolling(window=W_couple,min_periods=min_periods_couple).corr(df['dot_FEcrit_smooth'])
        else: df['CoupleIndex'] = 0.0
        df['CoupleIndex'] = df['CoupleIndex'].fillna(0) 
        return df

# --- Strain Profile Function (Reused) ---
def standard_shock_strain_profile(t, params, simulator_instance=None): # Renamed for clarity
    t_shock_start = params.get('t_shock_start', 50)
    t_shock_end = params.get('t_shock_end', t_shock_start + 10)
    if t_shock_end <= t_shock_start: t_shock_end = t_shock_start + 10.0001
    t_recovery_starts_strain = params.get('t_recovery_starts_strain', t_shock_end + 20)
    if t_recovery_starts_strain <= t_shock_end: t_recovery_starts_strain = t_shock_end + 20.0001
    strain_baseline = params.get('strain_baseline', 1.0)
    strain_shock_max = params.get('strain_shock_max', 10.0)
    strain_recovery_level = params.get('strain_recovery_level', 2.0)
    if t < t_shock_start: return strain_baseline
    elif t < t_shock_end:
        prog = (t-t_shock_start)/(t_shock_end-t_shock_start) if (t_shock_end-t_shock_start)>1e-6 else 1.0
        return strain_baseline + prog * (strain_shock_max - strain_baseline)
    elif t < t_recovery_starts_strain:
        prog = (t-t_shock_end)/(t_recovery_starts_strain-t_shock_end) if (t_recovery_starts_strain-t_shock_end)>1e-6 else 1.0
        return strain_shock_max - prog * (strain_shock_max - strain_recovery_level)
    else: return strain_recovery_level

# --- Plotting Functions (Reused robust versions) ---
# robust_plot_simulation_results and robust_plot_diagnostic_trajectories from previous response
def _add_phase_bands(ax, df, phase_colors):
    """Shade vertical bands that indicate the true phase."""
    ymin, ymax = ax.get_ylim()                # limits *after* plotting
    for phase_val, colour in phase_colors.items():
        if phase_val == -1 and not (df['current_phase_est_numeric'] == -1).any():
            continue
        ax.fill_between(df['t'], ymin, ymax,
                        where=df['current_phase_est_numeric'] == phase_val,
                        step='post', color=colour, alpha=0.35, zorder=-10)

def robust_plot_simulation_results(df, params, title_suffix=""):
    if df.empty or len(df) < 2:
        print(f"No data for '{title_suffix}'; skipping plot."); return

    fig, axs = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    phase_colors = {0:'lightgray',1:'salmon',2:'moccasin',
                    3:'lightgreen',4:'lightblue',-1:'white'}

    # ── 1. LEVERS ─────────────────────────────────────────────────────────
    axs[0].plot(df['t'], df['gLever'],  label='gLever')
    axs[0].plot(df['t'], df['betaLever'], label='betaLever')
    axs[0].plot(df['t'], df['FEcrit'],   label='FEcrit')
    axs[0].set_title('System Levers Over Time')
    axs[0].legend(); axs[0].grid(True, ls=':')

    # ── 2. STRAIN vs TOLERANCE ───────────────────────────────────────────
    axs[1].plot(df['t'], df['avg_delta_P_tau'], label='Strain')
    axs[1].plot(df['t'], df['ThetaT'],           label='ThetaT', ls='--')
    ymax = axs[1].get_ylim()[1]
    axs[1].plot(df['t'], df['is_collapsed']*ymax*0.95,
                label='Collapsed', ls=':', drawstyle='steps-post')
    axs[1].set_title('System Strain vs. Tolerance')
    axs[1].legend(); axs[1].grid(True, ls=':')

    # ── 3. DIAGNOSTICS ───────────────────────────────────────────────────
    ax3_t = axs[2].twinx()
    axs[2].plot(df['t'], df['SpeedIndex'],  label='SpeedIndex',  color='m')
    ax3_t.plot(df['t'], df['CoupleIndex'], label='CoupleIndex', color='darkcyan')
    axs[2].set_ylabel('SpeedIndex'); ax3_t.set_ylabel('CoupleIndex')
    axs[2].grid(True, ls=':'); axs[2].set_title('Speed & Couple Indices')
    lines1, labels1 = axs[2].get_legend_handles_labels()
    lines2, labels2 = ax3_t.get_legend_handles_labels()
    axs[2].legend(lines1+lines2, labels1+labels2)

    # ── 4. rhoE ───────────────────────────────────────────────────────────
    axs[3].plot(df['t'], df['rhoE'], label='rhoE', color='saddlebrown')
    axs[3].axhline(1.0, ls='--', color='gray', label='baseline')
    axs[3].set_title('Exploration Entropy Excess')
    axs[3].legend(); axs[3].grid(True, ls=':')

    # ── Shade the phase bands (do this LAST) ─────────────────────────────
    for ax in axs:
        _add_phase_bands(ax, df, phase_colors)

    plt.tight_layout(); plt.show()


def robust_plot_diagnostic_trajectories(df, title_suffix=""): # From previous, seems robust enough
    if df.empty or len(df) < 5: print(f"DataFrame for '{title_suffix}' is too short for trajectory plots."); return
    fig_traj=plt.figure(figsize=(16,7.5)); ax_2d=fig_traj.add_subplot(1,2,1); rhoE_plot=df['rhoE'].copy(); rhoE_plot_finite=rhoE_plot[np.isfinite(rhoE_plot)&(rhoE_plot>=0)]
    rhoE_cap_val=10.0
    if not rhoE_plot_finite.empty:
        q99=rhoE_plot_finite.quantile(0.99) if len(rhoE_plot_finite)>1 else rhoE_plot_finite.iloc[0]; current_max=rhoE_plot_finite.max(); current_min=rhoE_plot_finite.min()
        q99=q99 if not np.isnan(q99) else current_max
        if current_max<=1.0: rhoE_cap_val=max(1.5,current_max+0.5)
        else: rhoE_cap_val=max(q99,current_min+1.0,2.0)
        if np.isnan(rhoE_cap_val) or rhoE_cap_val==0: rhoE_cap_val=10.0
    rhoE_capped_for_color=np.clip(rhoE_plot.fillna(0),0,rhoE_cap_val); speed_max_val_plot=df['SpeedIndex'].max() if not df['SpeedIndex'].empty else 0
    xlim_left=(-0.1*speed_max_val_plot) if speed_max_val_plot>0 else -0.1
    sc=ax_2d.scatter(df['SpeedIndex'],df['CoupleIndex'],c=rhoE_capped_for_color,cmap='viridis',s=20,alpha=0.65,vmin=0,vmax=max(1.0,rhoE_cap_val))
    ax_2d.set_xlabel('SpeedIndex',fontsize=12); ax_2d.set_ylabel('CoupleIndex',fontsize=12); ax_2d.set_title('Trajectory in (SpeedIndex, CoupleIndex) Plane\nColor-coded by rhoE (capped)',fontsize=14)
    ax_2d.set_xlim(left=xlim_left); ax_2d.set_ylim(-1.15,1.15); cbar=plt.colorbar(sc,ax=ax_2d,aspect=30); cbar.set_label('rhoE (capped)',fontsize=12); ax_2d.grid(True,ls=':',alpha=0.7)
    ax_2d.plot(df['SpeedIndex'].iloc[0],df['CoupleIndex'].iloc[0],'o',color='lime',ms=10,label='Start',mec='k',zorder=5); ax_2d.plot(df['SpeedIndex'].iloc[-1],df['CoupleIndex'].iloc[-1],'X',color='r',ms=10,label='End',mec='k',zorder=5)
    ax_2d.plot(df['SpeedIndex'],df['CoupleIndex'],color='dimgray',ls='-',lw=0.8,alpha=0.5,zorder=4); ax_2d.legend(fontsize=10)
    ax_3d=fig_traj.add_subplot(1,2,2,projection='3d'); rhoE_plot_3d_capped=np.clip(df['rhoE'].fillna(0),0,rhoE_cap_val)
    sc_3d=ax_3d.scatter(df['SpeedIndex'],df['CoupleIndex'],rhoE_plot_3d_capped,c=df['t'],cmap='plasma',s=20,alpha=0.65)
    ax_3d.set_xlabel('SpeedIndex',fontsize=12); ax_3d.set_ylabel('CoupleIndex',fontsize=12); ax_3d.set_zlabel('rhoE (capped)',fontsize=12)
    ax_3d.set_title('Trajectory in (Speed, Couple, rhoE) Space\nColor-coded by Time',fontsize=14); ax_3d.set_zlim(0,max(1.0,rhoE_cap_val*1.05))
    cbar_3d=plt.colorbar(sc_3d,ax=ax_3d,pad=0.12,fraction=0.03,aspect=30); cbar_3d.set_label('Time (t)',fontsize=12)
    ax_3d.plot([df['SpeedIndex'].iloc[0]],[df['CoupleIndex'].iloc[0]],[rhoE_plot_3d_capped.iloc[0]],'o',color='lime',ms=10,label='Start',mec='k',zorder=10)
    ax_3d.plot([df['SpeedIndex'].iloc[-1]],[df['CoupleIndex'].iloc[-1]],[rhoE_plot_3d_capped.iloc[-1]],'X',color='r',ms=10,label='End',mec='k',zorder=10)
    ax_3d.plot(df['SpeedIndex'],df['CoupleIndex'],rhoE_plot_3d_capped,color='dimgray',ls='-',lw=0.8,alpha=0.5,zorder=1); ax_3d.legend(loc='upper left',bbox_to_anchor=(0.02,0.98),fontsize=10)
    plt.tight_layout(rect=[0,0,1,0.94]); fig_traj_title="Diagnostic Trajectories: "+title_suffix if title_suffix else "Diagnostic Trajectories"; plt.suptitle(fig_traj_title,fontsize=18,y=0.97); plt.show()


# --- Main Simulation Execution ---
if __name__ == "__main__":
    
    # Step 1: Run baseline simulations to establish EntropyBaseline for each method
    print("--- Establishing Entropy Baselines ---")
    baseline_params = DEFAULT_PARAMS_MULTI.copy() # Use the same name as the previous script
    baseline_params['T_max'] = 200 # Shorter run, forced stable
    baseline_params['strain_profile_func'] = lambda t, p, s: p.get('strain_baseline', 1.0) # Constant low strain
    # Force PRE_COLLAPSE mode for baseline calculation by manipulating targets
    baseline_params['beta_exploratory'] = baseline_params['beta_normal_op_target']
    baseline_params['g_exploratory'] = baseline_params['g_normal_op_target']


    entropy_baselines = {}
    entropy_methods_to_test = ['shannon', 'variance', 'range']

    for method in entropy_methods_to_test:
        print(f"  Running baseline for Entropy Method: {method}")
        current_baseline_params = baseline_params.copy()
        current_baseline_params['entropy_exp_method'] = method
        
        # Initialize ABM for baseline run
        abm_for_baseline = ABMSystem(
            current_baseline_params.get('num_abm_agents', 100),
            current_baseline_params.get('num_abm_activity_levels', 10)
        )
        
        simulator_baseline = PhoenixLoopSimulatorWithABM(current_baseline_params, abm_system_instance=abm_for_baseline)
        # Force PRE_COLLAPSE state for the whole duration for baseline
        simulator_baseline.current_phase_est = "PRE_COLLAPSE" 
        def force_pre_collapse_targets(sim_instance): # Override target setting for baseline
            sim_instance.betaLever = sim_instance.params['beta_normal_op_target']
            sim_instance.gLever = sim_instance.params['g_normal_op_target']
            return sim_instance.params['g_normal_op_target'], sim_instance.params['beta_normal_op_target'], sim_instance.params['F_base_cost_normal']
        original_set_phase_targets = simulator_baseline._set_phase_targets 
        simulator_baseline._set_phase_targets = lambda: force_pre_collapse_targets(simulator_baseline)


        df_baseline = simulator_baseline.run_simulation()
        simulator_baseline._set_phase_targets = original_set_phase_targets # Restore original method


        if not df_baseline.empty and 'EntropyExp' in df_baseline.columns and len(df_baseline) > 50 : # Ensure enough data points
            # Use median of the latter half of the stable run
            stable_entropy_values = df_baseline['EntropyExp'].iloc[len(df_baseline)//2:]
            
            # Test different baseline calculations
            entropy_baselines[method] = {
                'median': np.median(stable_entropy_values),
                'p10': np.percentile(stable_entropy_values, 10)
            }
            print(f"    Baselines for {method}: Median={entropy_baselines[method]['median']:.3f}, P10={entropy_baselines[method]['p10']:.3f}")
        else:
            print(f"    Warning: Not enough data to reliably calculate baseline for {method}. Using default 1.0.")
            entropy_baselines[method] = {'median': 1.0, 'p10': 1.0}


    # Step 2: Run Phoenix Loop simulations using different entropy methods and baselines
    print("\n--- Running Phoenix Loop Simulations with different RhoE calculations ---")
    
    # Use a standard shock scenario that is known to cause collapse and recovery
    phoenix_run_base_params = DEFAULT_PARAMS_MULTI.copy()
    phoenix_run_base_params['strain_profile_func'] = standard_shock_strain_profile
    phoenix_run_base_params['t_shock_start'] = 100.0
    phoenix_run_base_params['t_shock_end'] = phoenix_run_base_params['t_shock_start'] + 15.0
    phoenix_run_base_params['t_recovery_starts_strain'] = phoenix_run_base_params['t_shock_end'] + 40.0
    phoenix_run_base_params['strain_shock_max'] = 12.0 # Strong shock
    phoenix_run_base_params['F_influx_rate'] = 1.5 # Moderate influx to allow recovery but not too fast
    phoenix_run_base_params['FEcrit_min_abs'] = 0.05
    phoenix_run_base_params['T_max'] = 500 # Longer run to see full recovery

    for method in entropy_methods_to_test:
        for baseline_type in ['median', 'p10']:
            sim_title_suffix = f"EntropyMethod_{method}_BaselineType_{baseline_type}"
            print(f"\n  Running Phoenix Loop for: {sim_title_suffix}")
            
            current_sim_params = phoenix_run_base_params.copy()
            current_sim_params['entropy_exp_method'] = method
            current_sim_params['EntropyBaseline'] = entropy_baselines[method][baseline_type]
            if current_sim_params['EntropyBaseline'] < 1e-9: # Prevent division by zero if baseline is tiny
                print(f"    Warning: Baseline for {method}/{baseline_type} is near zero ({current_sim_params['EntropyBaseline']:.2e}). Setting to 1e-9 for rhoE calculation.")
                current_sim_params['EntropyBaseline'] = 1e-9


            # Each simulation needs its own ABM instance to reset agent states
            abm_for_run = ABMSystem(
                current_sim_params.get('num_abm_agents', 100),
                current_sim_params.get('num_abm_activity_levels', 10)
            )
            
            simulator_run = PhoenixLoopSimulatorWithABM(current_sim_params, abm_system_instance=abm_for_run)
            results_df_run = simulator_run.run_simulation()

            print(f"    Simulation for {sim_title_suffix} complete.")
            print(f"      System collapsed: {'Yes' if simulator_run.is_collapsed else 'No'}")
            if simulator_run.is_collapsed:
                print(f"      Collapse time: {simulator_run.collapse_time:.2f}")
            print(f"      Final estimated phase: {simulator_run.current_phase_est}")

            if results_df_run.empty or len(results_df_run) < 10:
                 print(f"      Warning: Results DataFrame is too short for {sim_title_suffix}. Skipping plots.")
                 continue

            print(f"\n    Plotting results for {sim_title_suffix}...")
            robust_plot_simulation_results(results_df_run, current_sim_params, title_suffix=sim_title_suffix)
            robust_plot_diagnostic_trajectories(results_df_run, title_suffix=sim_title_suffix)
            print(f"    Plotting for {sim_title_suffix} complete.")

    print("\n--- All entropy robustness scenarios processed. ---")

