#phoenix_loop_scenarios_sim.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # For custom legends
import mpl_toolkits.mplot3d  # For 3D plots
import matplotlib as mpl

FONT_SCALE = 2  # Double all text sizes in plots
plt.rcParams.update({
    'font.size': plt.rcParams.get('font.size', 10) * FONT_SCALE,
    'savefig.dpi': 350
})
import os
import json
import sys
from scipy.stats import pearsonr

# Glossary of variables used in the simulation.  Values are mostly
# unitless ("a.u." stands for arbitrary units) but are documented here so
# that downstream text-generation tools do not have to guess their
# meaning.
VARIABLE_GLOSSARY = {
    'gLever': {'units': 'a.u.', 'description': 'resource allocation lever'},
    'betaLever': {'units': 'a.u.', 'description': 'exploration/stochasticity lever'},
    'FEcrit': {'units': 'a.u.', 'description': 'critical free energy reserve'},
    'avg_delta_P_tau': {'units': 'a.u.', 'description': 'average accumulated strain'},
    'ThetaT': {'units': 'a.u.', 'description': 'tolerance threshold'},
    'rhoE': {'units': 'ratio', 'description': 'entropy relative to baseline'},
    'EntropyExp': {'units': 'a.u.', 'description': 'expected entropy (1/beta)'},
    'SpeedIndex': {'units': 'a.u./s', 'description': 'magnitude of change in beta and FEcrit'},
    'CoupleIndex': {'units': 'corr', 'description': 'rolling correlation between beta and FEcrit rates'},
    'safety_margin': {'units': 'a.u.', 'description': 'ThetaT minus avg_delta_P_tau'},
}

PHASE_RATIONALE = {
    'PRE_COLLAPSE': 'initial healthy operation before any collapse triggers',
    'DISINTEGRATION': 'immediate decay after collapse criterion met',
    'FLARING': 'high exploration to regain stability',
    'PRUNING': 'selective reduction of over-extended components',
    'RESTABILIZATION': 'return to steady operation once metrics recover'
}

THRESHOLD_INFO = {
    'rhoE_cross_high': 1.5,
    'couple_low': -0.5,
}

def get_version_info():
    """Collect versions of key packages and the Python interpreter."""
    return {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'matplotlib': mpl.__version__,
        'scipy': __import__('scipy').__version__,
    }
class PhoenixLoopSimulator:
    def __init__(self, params):
        self.params = params
        self.t = 0.0

        # Initialize state variables from params
        self.gLever = params['g0']
        self.betaLever = params['beta0']
        self.FEcrit = params['Fcrit0']
        
        _beta_for_baseline = params.get('EntropyExp_baseline_beta_val', params['beta_normal_op_target'])
        if _beta_for_baseline <= 1e-6: _beta_for_baseline = 1e-6
        self.EntropyExp_baseline = 1.0 / _beta_for_baseline

        # Pass 'self' to strain_profile_func if it needs access to simulator state
        self.avg_delta_P_tau = self.params['strain_profile_func'](self.t, params, self)
        self.ThetaT = self._calculate_ThetaT()

        self.dot_gLever = 0.0
        self.dot_betaLever = 0.0
        self.dot_FEcrit = 0.0
        
        self.EntropyExp = self._calculate_EntropyExp()
        self.rhoE = self._calculate_rhoE(self.EntropyExp)

        self.is_collapsed = False
        self.collapse_time = -1.0
        self.current_phase_est = "PRE_COLLAPSE"

        self.history = []
        self._record_history()

        # For scenario-specific logic
        self.fcrit_shock_applied = False 
        self.lever_cost_escalation_active = False 
        self.original_k_g = params.get('k_g', 0.1) 

    def _calculate_ThetaT(self):
        g = max(self.gLever, 1e-6)
        b = max(self.betaLever, 1e-6)
        f = max(self.FEcrit, 1e-6)
        return self.params['C'] * \
               (g ** self.params['w1']) * \
               (b ** self.params['w2']) * \
               (f ** self.params['w3'])

    def _calculate_lever_costs(self):
        current_k_g = self.params['k_g']
        if self.lever_cost_escalation_active: # This flag is controlled in step()
            current_k_g = self.params.get('k_g_escalated', self.params['k_g'] * 10) # Use escalated k_g
            
        cost_g = current_k_g * (self.gLever ** self.params['phi1'])
        cost_beta = self.params['k_b'] * (self.betaLever ** self.params['phi_b'])
        return cost_g, cost_beta

    def _update_FEcrit(self, cost_g, cost_beta, current_base_cost):
        dFEcrit_dt = self.params['F_influx_rate'] - current_base_cost - cost_g - cost_beta
        self.FEcrit += dFEcrit_dt * self.params['dt']
        self.FEcrit = max(0, self.FEcrit) 

    def _calculate_EntropyExp(self):
        if self.betaLever > 1e-6:
            return 1.0 / self.betaLever
        else:
            return 1.0 / 1e-6 

    def _calculate_rhoE(self, current_EntropyExp):
        if self.EntropyExp_baseline > 1e-6:
            return current_EntropyExp / self.EntropyExp_baseline
        return 1.0 

    def _set_phase_targets(self):
        target_g = self.gLever
        target_beta = self.betaLever
        current_base_cost = self.params['F_base_cost_normal']

        if self.current_phase_est == "PRE_COLLAPSE":
            target_g = self.params['g_normal_op_target']
            target_beta = self.params['beta_normal_op_target']
            if self.params.get('pre_collapse_drift_Fcrit_rate', 0) != 0:
                 self.FEcrit += self.params['pre_collapse_drift_Fcrit_rate'] * self.params['dt']
            if self.params.get('pre_collapse_drift_beta_rate', 0) != 0:
                 self.betaLever += self.params['pre_collapse_drift_beta_rate'] * self.params['dt']
        elif self.current_phase_est == "DISINTEGRATION":
            current_base_cost = self.params['F_base_cost_recovery'] 
            target_g = self.gLever 
            target_beta = self.betaLever
        elif self.current_phase_est == "FLARING":
            target_beta = self.params['beta_exploratory']
            if self.FEcrit < self.params['Fcrit_g_reduction_threshold']:
                target_g = self.params['g_exploratory_low_fcrit']
            else:
                target_g = self.params['g_exploratory']
            current_base_cost = self.params['F_base_cost_recovery']
        elif self.current_phase_est == "PRUNING":
            target_beta = self.params['beta_pruning_target']
            target_g = self.params['g_pruning_target']
            current_base_cost = self.params['F_base_cost_recovery'] 
        elif self.current_phase_est == "RESTABILIZATION":
            target_beta = self.params['beta_restabilization_target']
            target_g = self.params['g_restabilization_target']
            current_base_cost = self.params['F_base_cost_normal']
        return target_g, target_beta, current_base_cost

    def _update_phase_estimation(self):
        collapse_condition_met = False
        if self.FEcrit <= self.params['FEcrit_min_abs']: # Use <= for min_abs
            collapse_condition_met = True
        elif self.ThetaT > 1e-6 and (self.avg_delta_P_tau / self.ThetaT > self.params['strain_collapse_ratio']):
            collapse_condition_met = True
        elif self.params.get('beta_collapse_threshold', float('inf')) <= self.betaLever :
            collapse_condition_met = True

        if not self.is_collapsed and collapse_condition_met:
            self.is_collapsed = True
            self.collapse_time = self.t
            self.current_phase_est = "DISINTEGRATION"
        
        if self.is_collapsed:
            time_since_collapse_start = self.t - self.collapse_time
            disintegration_actual_duration = self.params.get('disintegration_duration', self.params['dt'])

            if self.current_phase_est == "DISINTEGRATION":
                if time_since_collapse_start >= disintegration_actual_duration:
                    self.current_phase_est = "FLARING"
            
            elif self.current_phase_est == "FLARING":
                time_in_flaring = time_since_collapse_start - disintegration_actual_duration
                if time_in_flaring >= self.params['min_flaring_duration'] and \
                   self.FEcrit >= self.params['Fcrit_pruning_threshold']:
                    self.current_phase_est = "PRUNING"
            
            elif self.current_phase_est == "PRUNING":
                time_in_pruning = time_since_collapse_start - disintegration_actual_duration - self.params['min_flaring_duration']
                beta_stabilized_enough = abs(self.betaLever - self.params['beta_pruning_target']) < 0.20 * self.params['beta_pruning_target'] 
                if time_in_pruning >= self.params['min_pruning_duration'] and \
                   self.FEcrit >= self.params['Fcrit_restabilization_threshold'] and beta_stabilized_enough :
                    self.current_phase_est = "RESTABILIZATION"

    def step(self):
        # Fcrit Shock Logic
        t_fcrit_shock = self.params.get('t_fcrit_shock', -1)
        if t_fcrit_shock >= 0 and abs(self.t - t_fcrit_shock) < self.params['dt']/2 and not self.fcrit_shock_applied:
            shock_val = self.params.get('fcrit_shock_abs_drop', 0)
            self.FEcrit = max(0, self.FEcrit - shock_val) # Ensure Fcrit doesn't go below 0 from shock itself
            self.fcrit_shock_applied = True

        # Lever Cost Escalation/Restoration Logic
        t_lever_cost_esc_start = self.params.get('t_lever_cost_esc_start', -1)
        t_lever_cost_esc_end = self.params.get('t_lever_cost_esc_end', -1)
        
        if t_lever_cost_esc_start >= 0 and self.t >= t_lever_cost_esc_start and (t_lever_cost_esc_end < 0 or self.t < t_lever_cost_esc_end) :
             if not self.lever_cost_escalation_active: # Activate escalation
                self.lever_cost_escalation_active = True
        elif self.lever_cost_escalation_active and (t_lever_cost_esc_end >= 0 and self.t >= t_lever_cost_esc_end):
            self.lever_cost_escalation_active = False # Deactivate escalation

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
        self.EntropyExp = self._calculate_EntropyExp()
        self.rhoE = self._calculate_rhoE(self.EntropyExp)
        self._record_history()
        self.t += self.params['dt']

    def _record_history(self):
        phase_numeric_map = {"PRE_COLLAPSE": 0, "DISINTEGRATION": 1, "FLARING": 2, "PRUNING": 3, "RESTABILIZATION": 4}
        self.history.append({
            't': self.t,
            'gLever': self.gLever,
            'betaLever': self.betaLever,
            'FEcrit': self.FEcrit,
            'avg_delta_P_tau': self.avg_delta_P_tau,
            'ThetaT': self.ThetaT,
            'is_collapsed': int(self.is_collapsed),
            'current_phase_est_numeric': phase_numeric_map.get(self.current_phase_est, -1), 
            'rhoE': self.rhoE,
            'EntropyExp': self.EntropyExp,
        })

    def run_simulation(self):
        self.fcrit_shock_applied = False
        self.lever_cost_escalation_active = False
        # self.params['k_g'] = self.original_k_g # Reset k_g. Done in constructor copy now.

        while self.t < self.params['T_max']:
            if self.is_collapsed and self.current_phase_est == "RESTABILIZATION":
                time_in_restabilization = self.t - (
                    self.collapse_time + 
                    self.params.get('disintegration_duration', self.params['dt']) +
                    self.params['min_flaring_duration'] + 
                    self.params['min_pruning_duration']
                )
                if time_in_restabilization > self.params.get("stabilization_duration_check", 50):
                    break 
            self.step()
        
        df = pd.DataFrame(self.history)
        if df.empty: # Handle case where simulation ends very quickly (e.g. instant collapse)
            return df

        df['dot_gLever'] = df['gLever'].diff() / self.params['dt']
        df['dot_betaLever'] = df['betaLever'].diff() / self.params['dt']
        df['dot_FEcrit'] = df['FEcrit'].diff() / self.params['dt']
        
        for col in ['dot_gLever', 'dot_betaLever', 'dot_FEcrit']:
            df[col] = df[col].fillna(0)

        smooth_window = self.params.get('W_smooth_deriv', 1)
        if smooth_window > 1 and len(df) >= smooth_window : # Check if df is long enough
            df['dot_betaLever_smooth'] = df['dot_betaLever'].rolling(window=smooth_window, center=True, min_periods=1).mean()
            df['dot_FEcrit_smooth'] = df['dot_FEcrit'].rolling(window=smooth_window, center=True, min_periods=1).mean()
        else:
            df['dot_betaLever_smooth'] = df['dot_betaLever']
            df['dot_FEcrit_smooth'] = df['dot_FEcrit']
            
        df['SpeedIndex'] = np.sqrt(df['dot_betaLever_smooth']**2 + df['dot_FEcrit_smooth']**2)
        
        W_couple = self.params.get('W_couple', 20)
        min_periods_couple = max(2, W_couple // 2) 
        if len(df) >= min_periods_couple: # Check if df is long enough for correlation window
            df['CoupleIndex'] = df['dot_betaLever_smooth'].rolling(window=W_couple, min_periods=min_periods_couple).corr(df['dot_FEcrit_smooth'])
        else:
            df['CoupleIndex'] = 0.0 # Not enough data for correlation
        df['CoupleIndex'] = df['CoupleIndex'].fillna(0) 

        return df

# --- Strain Profile Functions for different scenarios ---
def shock_strain_profile(t, params, simulator_instance=None): 
    t_shock_start = params.get('t_shock_start', 50)
    t_shock_end = params.get('t_shock_end', t_shock_start + 10)
    if t_shock_end <= t_shock_start: t_shock_end = t_shock_start + 10.0001 # Ensure duration
    
    t_recovery_starts_strain = params.get('t_recovery_starts_strain', t_shock_end + 20)
    if t_recovery_starts_strain <= t_shock_end: t_recovery_starts_strain = t_shock_end + 20.0001


    strain_baseline = params.get('strain_baseline', 1.0)
    strain_shock_max = params.get('strain_shock_max', 10.0)
    strain_recovery_level = params.get('strain_recovery_level', 2.0)

    if t < t_shock_start:
        return strain_baseline
    elif t < t_shock_end:
        progress = (t - t_shock_start) / (t_shock_end - t_shock_start) if (t_shock_end - t_shock_start) > 1e-6 else 1.0
        return strain_baseline + progress * (strain_shock_max - strain_baseline)
    elif t < t_recovery_starts_strain:
        progress = (t - t_shock_end) / (t_recovery_starts_strain - t_shock_end) if (t_recovery_starts_strain - t_shock_end) > 1e-6 else 1.0
        return strain_shock_max - progress * (strain_shock_max - strain_recovery_level)
    else:
        return strain_recovery_level

def sustained_high_strain_profile(t, params, simulator_instance=None):
    t_rise_start = params.get('t_strain_rise_start', 30)
    t_rise_end = params.get('t_strain_rise_end', t_rise_start + 20)
    if t_rise_end <= t_rise_start: t_rise_end = t_rise_start + 20.0001 

    strain_baseline = params.get('strain_baseline', 2.0)
    strain_high_level = params.get('strain_high_sustained', 7.0) 

    if t < t_rise_start:
        return strain_baseline
    elif t < t_rise_end:
        progress = (t - t_rise_start) / (t_rise_end - t_rise_start) if (t_rise_end - t_rise_start) > 1e-6 else 1.0
        return strain_baseline + progress * (strain_high_level - strain_baseline)
    else:
        return strain_high_level 

def normal_strain_for_internal_collapse(t, params, simulator_instance=None):
    return params.get('strain_baseline', 2.0)


# --- Default Parameters ---
DEFAULT_PARAMS_MULTI = {
    'g0': 1.0, 'beta0': 2.0, 'Fcrit0': 100.0,
    'C': 1.0, 'w1': 0.33, 'w2': 0.33, 'w3': 0.34,
    'k_g': 0.1, 'phi1': 0.5, 'k_b': 0.05, 'phi_b': 1.0,
    'F_influx_rate': 2.0, 
    'F_base_cost_normal': 1.0, 
    'F_base_cost_recovery': 0.3, 
    'FEcrit_min_abs': 0.1, # Lowered for more definitive collapse
    'strain_collapse_ratio': 1.05, # Slightly more sensitive
    'dt': 0.1, 'T_max': 450.0, # Increased T_max
    'W_couple': 50, 
    'W_smooth_deriv': 5,

    'g_adaptation_rate': 0.2, 'beta_adaptation_rate': 0.1,
    'g_min': 0.05, 'g_max': 5.0, 'beta_min': 0.01, 'beta_max': 20.0,

    'g_normal_op_target': 1.0, 'beta_normal_op_target': 2.0,
    'EntropyExp_baseline_beta_val': 2.0,

    'disintegration_duration': 2.0, 
    
    'beta_exploratory': 0.05, 
    'g_exploratory': 0.7,      
    'g_exploratory_low_fcrit': 0.2, 
    'Fcrit_g_reduction_threshold': 15.0,
    
    'min_flaring_duration': 70.0, # Increased flaring 
    'Fcrit_pruning_threshold': 35.0, 
    
    'beta_pruning_target': 1.5,
    'g_pruning_target': 1.2,
    
    'min_pruning_duration': 70.0, # Increased pruning
    'Fcrit_restabilization_threshold': 65.0,
    'beta_restabilization_target': 1.8, 
    'g_restabilization_target': 1.0,  
    'stabilization_duration_check': 80.0, 
    
    'strain_profile_func': shock_strain_profile, 
    'strain_baseline': 2.0, 
}
def _add_phase_bands(ax, df, phase_colors):
    ymin, ymax = ax.get_ylim()                  # limits *after* data are drawn
    for phase_val, colour in phase_colors.items():
        if phase_val == -1 and not (df['current_phase_est_numeric'] == -1).any():
            continue
        ax.fill_between(df['t'], ymin, ymax,
                        where=df['current_phase_est_numeric'] == phase_val,
                        step='post', color=colour, alpha=0.35, zorder=-10)

def robust_plot_simulation_results(df, params, title_suffix=""):
    if df.empty or len(df) < 2:
        print(f"No data for '{title_suffix}'; skipping plot."); return

    # The function creates a figure object named 'fig' here
    fig, axs = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    phase_colors = {0:'lightgray',1:'salmon',2:'moccasin',3:'lightgreen',4:'lightblue',-1:'white'}

    # 1 ── Levers -----------------------------------------------------------
    axs[0].plot(df['t'], df['gLever'],  label='gLever')
    axs[0].plot(df['t'], df['betaLever'], label='betaLever')
    axs[0].plot(df['t'], df['FEcrit'],   label='FEcrit')
    axs[0].set_title('System Levers Over Time'); axs[0].legend(); axs[0].grid(True, ls=':')

    # 2 ── Strain vs. tolerance -------------------------------------------
    axs[1].plot(df['t'], df['avg_delta_P_tau'], label='Strain')
    axs[1].plot(df['t'], df['ThetaT'],           label='ThetaT', ls='--')
    # Use try-except to handle potential empty ylim issue if plots are empty
    try:
        ymax = axs[1].get_ylim()[1]
        axs[1].plot(df['t'], df['is_collapsed']*ymax*0.95,
                    label='Collapsed', ls=':', drawstyle='steps-post')
    except IndexError:
        pass # Skip plotting the collapse line if ylim is not set
    axs[1].set_title('System Strain vs. Tolerance'); axs[1].legend(); axs[1].grid(True, ls=':')

    # 3 ── Diagnostics -----------------------------------------------------
    ax3_t = axs[2].twinx()
    axs[2].plot(df['t'], df['SpeedIndex'],  label='SpeedIndex',  color='m')
    ax3_t.plot(df['t'], df['CoupleIndex'], label='CoupleIndex', color='darkcyan')
    axs[2].set_ylabel('SpeedIndex'); ax3_t.set_ylabel('CoupleIndex')
    axs[2].grid(True, ls=':'); axs[2].set_title('Speed & Couple Indices')
    lines1, labels1 = axs[2].get_legend_handles_labels()
    lines2, labels2 = ax3_t.get_legend_handles_labels()
    axs[2].legend(lines1+lines2, labels1+labels2)

    # 4 ── rhoE ------------------------------------------------------------
    axs[3].plot(df['t'], df['rhoE'], label='rhoE', color='saddlebrown')
    axs[3].axhline(1.0, ls='--', color='gray', label='baseline')
    axs[3].set_title('Exploration Entropy Excess'); axs[3].legend(); axs[3].grid(True, ls=':')

    # ------ add background bands (do it last so autoscaling is finished) --
    for ax in axs:
        _add_phase_bands(ax, df, phase_colors)

    # --- CORRECTED SECTION: SAVE THE FIGURE ---
    plt.tight_layout()
    
    # Add a title to the whole figure
    fig_title = "System Dynamics Time Series: " + title_suffix if title_suffix else "System Dynamics Time Series"
    fig.suptitle(fig_title, fontsize=36, y=1.02)  # Use fig.suptitle for figure-level title
    plt.subplots_adjust(top=0.96) # Adjust layout to make room for the title

    # Construct the save path and save the figure
    save_path = os.path.join("results", f"sim_{title_suffix}_time_series.png")
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    
    # Close the figure to free up memory
    plt.close(fig)

def robust_plot_diagnostic_trajectories(df, title_suffix=""):
    if df.empty or len(df) < 5: # Need a few points for trajectories
        print(f"DataFrame for '{title_suffix}' is too short or empty for trajectory plots.")
        return
        
    fig_traj = plt.figure(figsize=(16, 7.5)) 

    # 2D Plot
    ax_2d = fig_traj.add_subplot(1, 2, 1)
    rhoE_plot = df['rhoE'].copy()
    rhoE_plot_finite = rhoE_plot[np.isfinite(rhoE_plot) & (rhoE_plot >= 0)]
    
    rhoE_cap_val = 10.0 # Default cap
    if not rhoE_plot_finite.empty:
        q99 = rhoE_plot_finite.quantile(0.99) if len(rhoE_plot_finite) > 1 else rhoE_plot_finite.iloc[0]
        current_max = rhoE_plot_finite.max() if not rhoE_plot_finite.empty else 0
        current_min = rhoE_plot_finite.min() if not rhoE_plot_finite.empty else 0
        
        # Ensure q99 is sensible
        q99 = q99 if not np.isnan(q99) else current_max

        # Determine a reasonable cap value
        if current_max <= 1.0: # If all rhoE values are low
            rhoE_cap_val = max(1.5, current_max + 0.5) # Ensure cap is at least 1.5
        else:
            rhoE_cap_val = max(q99, current_min + 1.0, 2.0) # Cap should be at least 2 if there are higher values
        if np.isnan(rhoE_cap_val) or rhoE_cap_val == 0: rhoE_cap_val = 10.0 # Fallback

    rhoE_capped_for_color = np.clip(rhoE_plot.fillna(0), 0, rhoE_cap_val)

    # Ensure SpeedIndex for xlim is not problematic
    speed_max_val_plot = df['SpeedIndex'].max() if not df['SpeedIndex'].empty else 0
    xlim_left = (-0.1 * speed_max_val_plot) if speed_max_val_plot > 0 else -0.1


    sc = ax_2d.scatter(df['SpeedIndex'], df['CoupleIndex'], c=rhoE_capped_for_color, cmap='viridis', s=20, alpha=0.65, vmin=0, vmax=max(1.0, rhoE_cap_val)) # Ensure vmax is at least 1
    ax_2d.set_xlabel('SpeedIndex', fontsize=24); ax_2d.set_ylabel('CoupleIndex', fontsize=24)
    ax_2d.set_title('Trajectory (SpeedIndex, CoupleIndex)', fontsize=26)
    ax_2d.set_xlim(left=xlim_left) 
    ax_2d.set_ylim(-1.15, 1.15); cbar = plt.colorbar(sc, ax=ax_2d, aspect=30)
    cbar.set_label('rhoE (capped)', fontsize=24); ax_2d.grid(True, linestyle=':', alpha=0.7)
    
    ax_2d.plot(df['SpeedIndex'].iloc[0], df['CoupleIndex'].iloc[0], 'o', color='lime', markersize=10, label='Start', markeredgecolor='black', zorder=5)
    ax_2d.plot(df['SpeedIndex'].iloc[-1], df['CoupleIndex'].iloc[-1], 'X', color='red', markersize=10, label='End', markeredgecolor='black', zorder=5)
    ax_2d.plot(df['SpeedIndex'], df['CoupleIndex'], color='dimgray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=4) 
    ax_2d.legend(fontsize=20)

    # 3D Plot
    ax_3d = fig_traj.add_subplot(1, 2, 2, projection='3d')
    rhoE_plot_3d_capped = np.clip(df['rhoE'].fillna(0), 0, rhoE_cap_val) 
    
    sc_3d = ax_3d.scatter(df['SpeedIndex'], df['CoupleIndex'], rhoE_plot_3d_capped, 
                          c=df['t'], cmap='plasma', s=20, alpha=0.65) 
    ax_3d.set_xlabel('SpeedIndex', fontsize=16); ax_3d.set_ylabel('CoupleIndex', fontsize=16)
    ax_3d.set_zlabel('rhoE (capped)', fontsize=16)
    ax_3d.set_title('Trajectory (Speed, Couple, rhoE)', fontsize=26)
    ax_3d.tick_params(axis='both', which='major', labelsize=12)
    ax_3d.zaxis.set_tick_params(labelsize=12)

    ax_3d.set_zlim(0, max(1.0, rhoE_cap_val * 1.05)) # Ensure zlim is reasonable

    cbar_3d = plt.colorbar(sc_3d, ax=ax_3d, pad=0.12, fraction=0.03, aspect=30) 
    cbar_3d.set_label('Time (t)', fontsize=24)
    
    ax_3d.plot([df['SpeedIndex'].iloc[0]], [df['CoupleIndex'].iloc[0]], [rhoE_plot_3d_capped.iloc[0]],
               'o', color='lime', markersize=10, label='Start', markeredgecolor='black', zorder=10)
    ax_3d.plot([df['SpeedIndex'].iloc[-1]], [df['CoupleIndex'].iloc[-1]], [rhoE_plot_3d_capped.iloc[-1]],
               'X', color='red', markersize=10, label='End', markeredgecolor='black', zorder=10)
    ax_3d.plot(df['SpeedIndex'], df['CoupleIndex'], rhoE_plot_3d_capped, 
               color='dimgray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1) 
    ax_3d.legend(loc='upper left', bbox_to_anchor=(0.02,0.98), fontsize=20)

    plt.tight_layout(rect=[0,0,1,0.94]) 
    fig_traj_title = "Diagnostic Trajectories: " + title_suffix if title_suffix else "Diagnostic Trajectories"
    plt.suptitle(fig_traj_title, fontsize=36, y=0.97)

    save_path = os.path.join("results", f"sim_{title_suffix}_trajectory.png")
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    plt.close()

def compute_simulation_summary(df, simulator, scenario_name):
    """Return structured summary information for a single scenario."""
    phase_map = {0: 'PRE_COLLAPSE', 1: 'DISINTEGRATION', 2: 'FLARING',
                 3: 'PRUNING', 4: 'RESTABILIZATION'}

    summary = {
        'scenario_name': scenario_name,
        'collapsed': bool(simulator.is_collapsed),
        't_collapse': float(simulator.collapse_time),
        'baseline_entropy_method': 'beta_inverse',
        'baseline_value': float(simulator.EntropyExp_baseline),
        'entropy_method': 'beta_inverse',
        'sampling_rate_hz': float(1.0 / simulator.params['dt']),
        'aggregation': {
            'mean': 'arithmetic',
            'std': 'sample',
            'nan_policy': 'omit'
        },
        'collapse_criteria': {
            'FEcrit_min_abs': float(simulator.params['FEcrit_min_abs']),
            'strain_ratio': float(simulator.params['strain_collapse_ratio']),
            'beta_threshold': float(simulator.params.get('beta_collapse_threshold', float('inf')))
        },
        'variable_glossary': VARIABLE_GLOSSARY,
        'phase_rationale': PHASE_RATIONALE,
        'thresholds': THRESHOLD_INFO,
        'version_info': get_version_info(),
        'domain_context': 'purely computational simulation',
        'random_seed': None
    }

    if df.empty:
        return summary

    df = df.copy()
    df['safety_margin'] = df['ThetaT'] - df['avg_delta_P_tau']

    # Phase windows and durations
    phase_windows = {}
    phase_durations = {}
    phase_steps = {}
    current_phase = df['current_phase_est_numeric'].iloc[0]
    start_time = df['t'].iloc[0]
    for idx in range(1, len(df)):
        val = df['current_phase_est_numeric'].iloc[idx]
        if val != current_phase:
            end_time = df['t'].iloc[idx-1]
            phase_windows[phase_map[current_phase]] = [float(start_time), float(end_time)]
            phase_durations[phase_map[current_phase]] = float(end_time - start_time)
            phase_steps[phase_map[current_phase]] = int(df[(df['t'] >= start_time) & (df['t'] <= end_time)].shape[0])
            current_phase = val
            start_time = df['t'].iloc[idx]
    # last phase
    end_time = df['t'].iloc[-1]
    phase_windows[phase_map[current_phase]] = [float(start_time), float(end_time)]
    phase_durations[phase_map[current_phase]] = float(end_time - start_time)
    phase_steps[phase_map[current_phase]] = int(df[(df['t'] >= start_time) & (df['t'] <= end_time)].shape[0])

    summary['phase_windows'] = phase_windows
    summary['phase_durations'] = phase_durations
    summary['phase_steps'] = phase_steps

    metrics = ['SpeedIndex', 'CoupleIndex', 'rhoE', 'gLever', 'betaLever',
               'FEcrit', 'avg_delta_P_tau', 'ThetaT', 'safety_margin']
    stats = {}
    times_of_peaks = {}
    corr_per_phase = {}
    for phase_name, window in phase_windows.items():
        mask = (df['t'] >= window[0]) & (df['t'] <= window[1])
        phase_stats = {}
        phase_times = {}
        for col in metrics:
            series = df.loc[mask, col]
            if series.empty:
                continue
            descr = series.agg(['max', 'min', 'mean', 'std']).to_dict()
            phase_stats[col] = {k: float(v) if pd.notna(v) else None for k, v in descr.items()}
            idxmax = series.idxmax()
            phase_times[col] = float(df.loc[idxmax, 't']) if not series.empty else None
        stats[phase_name] = phase_stats
        times_of_peaks[phase_name] = phase_times
        # correlation between dot_betaLever_smooth and dot_FEcrit_smooth
        b_series = df.loc[mask, 'dot_betaLever_smooth']
        f_series = df.loc[mask, 'dot_FEcrit_smooth']
        if len(b_series) > 1 and len(f_series) > 1:
            corr, _ = pearsonr(b_series, f_series)
            corr_per_phase[phase_name] = float(corr)
        else:
            corr_per_phase[phase_name] = None

    summary['stats'] = stats
    summary['times_of_peaks'] = times_of_peaks
    summary['corr_dot_beta_dot_fcrit'] = corr_per_phase

    # crossings
    crossings = {}
    cond_rho = df['rhoE'] > 1.5
    crossings['rhoE>1.5'] = {
        'count': int(((cond_rho.astype(int).diff() == 1).sum())),
        'first': float(df.loc[cond_rho, 't'].iloc[0]) if cond_rho.any() else None
    }
    cond_couple = df['CoupleIndex'] < -0.5
    crossings['CoupleIndex<-0.5'] = {
        'count': int(((cond_couple.astype(int).diff() == 1).sum())),
        'first': float(df.loc[cond_couple, 't'].iloc[0]) if cond_couple.any() else None
    }
    summary['crossings'] = crossings

    # trajectory metrics
    bbox = {}
    for col in ['SpeedIndex', 'CoupleIndex', 'rhoE']:
        bbox[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
    summary['trajectory_bbox'] = bbox
    coords = df[['SpeedIndex', 'CoupleIndex', 'rhoE']].values
    if len(coords) > 1:
        path_len = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
    else:
        path_len = 0.0
    summary['path_length_3d'] = path_len

    final_cols = ['gLever', 'betaLever', 'FEcrit', 'rhoE', 'SpeedIndex', 'CoupleIndex']
    final_state = {col: float(df[col].iloc[-1]) for col in final_cols}
    start_state = {col: float(df[col].iloc[0]) for col in final_cols}
    summary['final_state'] = final_state
    summary['start_state'] = start_state

    trend_desc = {}
    for col in final_cols:
        delta = final_state[col] - start_state[col]
        base = abs(start_state[col]) if abs(start_state[col]) > 1e-9 else 1.0
        if abs(delta) < 0.05 * base:
            trend = 'stable'
        elif delta > 0:
            trend = 'increase'
        else:
            trend = 'decrease'
        trend_desc[col] = trend
    summary['trend_overall'] = trend_desc

    return summary
# --- Main Simulation Execution for Multiple Scenarios (Improved) ---
if __name__ == "__main__":
    scenarios = {}
    os.makedirs("results", exist_ok=True)
    # Scenario 1: Original Shock-Induced Collapse (strain shock)
    params_strain_shock = DEFAULT_PARAMS_MULTI.copy()
    params_strain_shock['strain_profile_func'] = shock_strain_profile
    params_strain_shock['t_shock_start'] = 80.0 
    params_strain_shock['t_shock_end'] = params_strain_shock['t_shock_start'] + 15.0
    params_strain_shock['t_recovery_starts_strain'] = params_strain_shock['t_shock_end'] + 30.0
    params_strain_shock['strain_shock_max'] = 10.0 
    params_strain_shock['FEcrit_min_abs'] = 0.05 
    params_strain_shock['T_max'] = 400 
    scenarios['Strain_Shock_Collapse'] = params_strain_shock

    # Scenario 2: FEcrit Shock Collapse (Made More Severe)
    params_fcrit_shock = DEFAULT_PARAMS_MULTI.copy()
    params_fcrit_shock['strain_profile_func'] = normal_strain_for_internal_collapse 
    params_fcrit_shock['t_fcrit_shock'] = 80.0  
    params_fcrit_shock['fcrit_shock_abs_drop'] = 99.9 # From 100 to 0.1
    params_fcrit_shock['F_influx_rate'] = 0.8 # Significantly Reduced influx
    params_fcrit_shock['FEcrit_min_abs'] = 0.5 # Shock takes it below this target
    params_fcrit_shock['T_max'] = 400
    scenarios['FEcrit_Shock_Collapse_Severe'] = params_fcrit_shock
    
    # Scenario 3: Sustained High Strain Collapse (Made More Severe)
    params_sust_strain = DEFAULT_PARAMS_MULTI.copy()
    params_sust_strain['strain_profile_func'] = sustained_high_strain_profile
    params_sust_strain['t_strain_rise_start'] = 60.0
    params_sust_strain['t_strain_rise_end'] = params_sust_strain['t_strain_rise_start'] + 40.0 
    params_sust_strain['strain_high_sustained'] = 12.0 
    params_sust_strain['F_influx_rate'] = 0.7 # Drastically reduced influx
    params_sust_strain['k_g'] = 0.20 # Increased cost of g 
    params_sust_strain['k_b'] = 0.10 # Increased cost of b
    params_sust_strain['FEcrit_min_abs'] = 0.5 
    params_sust_strain['T_max'] = 500 # Longer time for this potentially slower collapse
    scenarios['Sustained_High_Strain_Collapse_Severe'] = params_sust_strain

    # Scenario 4: Lever Cost Instability (Made More Severe)
    params_lever_inst = DEFAULT_PARAMS_MULTI.copy()
    params_lever_inst['strain_profile_func'] = normal_strain_for_internal_collapse 
    params_lever_inst['t_lever_cost_esc_start'] = 70.0
    params_lever_inst['t_lever_cost_esc_end'] = params_lever_inst['t_lever_cost_esc_start'] + 60.0 # Longer escalation
    params_lever_inst['k_g_escalated'] = DEFAULT_PARAMS_MULTI['k_g'] * 100 # Very drastic cost increase
    params_lever_inst['g_normal_op_target'] = 1.8 # Ensure g is actively used to feel the cost
    params_lever_inst['F_influx_rate'] = 1.0 # Significantly reduced influx
    params_lever_inst['FEcrit_min_abs'] = 0.2
    params_lever_inst['T_max'] = 450
    scenarios['Lever_Cost_Instability_Collapse_Severe'] = params_lever_inst
    
    # Scenario 5: Beta Runaway Collapse (Kept similar, as it worked)
    params_beta_runaway = DEFAULT_PARAMS_MULTI.copy()
    params_beta_runaway['strain_profile_func'] = normal_strain_for_internal_collapse
    params_beta_runaway['beta_normal_op_target'] = 18.0 # Slightly higher target
    params_beta_runaway['beta_adaptation_rate'] = 0.45 
    params_beta_runaway['k_b'] = 0.30 # Higher cost for beta
    params_beta_runaway['beta_collapse_threshold'] = 19.5 
    params_beta_runaway['FEcrit_min_abs'] = 0.05 
    params_beta_runaway['F_influx_rate'] = 2.0 # Reduced slightly
    params_beta_runaway['T_max'] = 400 
    scenarios['Beta_Runaway_Collapse'] = params_beta_runaway


    summaries = []
    scenario_names = list(scenarios.keys())
    baseline_name = scenario_names[0] if scenario_names else None
    baseline_final = None

    for sc_name, sc_params in scenarios.items():
        print(f"\n--- Running Scenario: {sc_name} ---")
        simulator = PhoenixLoopSimulator(sc_params.copy())
        results_df = simulator.run_simulation()

        print(f"Simulation for {sc_name} complete.")
        print(f"  System collapsed: {'Yes' if simulator.is_collapsed else 'No'}")
        if simulator.is_collapsed:
            print(f"  Collapse time: {simulator.collapse_time:.2f}")
        print(f"  Final estimated phase: {simulator.current_phase_est}")
        
        if results_df.empty:
            print(f"  Warning: Results DataFrame is empty for {sc_name}.")
        else:
            print(f"  Results shape: {results_df.shape}")
            for col in ['SpeedIndex', 'CoupleIndex', 'rhoE']:
                if results_df[col].isnull().any():
                    print(f"  Warning: NaN values found in {col} for {sc_name}. Sum: {results_df[col].isnull().sum()}")

        print(f"\n  Plotting results for {sc_name}...")
        robust_plot_simulation_results(results_df, sc_params, title_suffix=sc_name)
        robust_plot_diagnostic_trajectories(results_df, title_suffix=sc_name)
        print(f"  Plotting for {sc_name} complete.")

        summary = compute_simulation_summary(results_df, simulator, sc_name)
        if baseline_final is None:
            baseline_final = summary.get('final_state', {})
        else:
            deltas = {}
            for k, v in summary.get('final_state', {}).items():
                base_v = baseline_final.get(k)
                if base_v is not None:
                    deltas[k] = v - base_v
            summary['final_state_delta_from_baseline'] = deltas
        summaries.append(summary)

    summary_path = os.path.join("results", "summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"Summaries written to {summary_path}")
    except Exception as e:
        print(f"Error writing summary file: {e}")

    print("\n--- All scenarios processed. ---")

