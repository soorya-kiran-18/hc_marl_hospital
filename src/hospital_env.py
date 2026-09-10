import numpy as np
import pandas as pd

class HospitalEnv:
    def __init__(self, data_path="data/processed/train.csv"):
        print("Initializing Hospital Simulation Engine...")
        try:
            self.patients_df = pd.read_csv(data_path)
            self.total_patients = len(self.patients_df)
        except Exception as e:
            print(f"Warning: Could not load {data_path}. Using synthetic patient flow.")
            self.total_patients = 1000
            
        # Hardcoded constraints per the methodology[cite: 2]
        self.er_ratio_limit = 4      # 1 nurse : 4 patients
        self.icu_ratio_limit = 2     # 1 nurse : 2 patients
        self.ward_ratio_limit = 6    # 1 nurse : 6 patients
        self.total_float_nurses = 6  # Flexible pool for Meta-Agent[cite: 2]
        
        # Base permanent staffing
        self.er_nurses_base = 10
        self.icu_nurses_base = 5
        self.ward_nurses_base = 15
        
        self.time_step_mins = 15
        self.reset()

    def reset(self):
        self.current_step = 0
        self.er_census = 0
        self.icu_census = 0
        self.ward_census = 0
        
        # Tracks wait times for the equity penalty formula[cite: 2]
        self.wait_times = {'Private': [], 'Medicaid': [], 'Medicare': [], 'SelfPay': [], 'Other/Unknown': []}
        return self._get_obs()

    def _get_obs(self):
        # Outputs the current global state for the Meta-Agent[cite: 2]
        # Format: [ER Patients, ICU Patients, Ward Patients, Available Float Nurses]
        return np.array([
            self.er_census, 
            self.icu_census, 
            self.ward_census, 
            self.total_float_nurses
        ], dtype=np.float32)

    def step(self, meta_action):
        """
        meta_action: array of size 3 representing float nurse distribution [ER, ICU, Ward][cite: 2]
        e.g., [3, 1, 2] means 3 nurses to ER, 1 to ICU, 2 to Ward.
        """
        # Enforce conservation of nurses
        meta_action = np.clip(meta_action, 0, self.total_float_nurses)
        if sum(meta_action) > self.total_float_nurses:
            meta_action = (np.array(meta_action) / sum(meta_action)) * self.total_float_nurses
            meta_action = np.floor(meta_action).astype(int)

        current_er_nurses = self.er_nurses_base + meta_action[0]
        current_icu_nurses = self.icu_nurses_base + meta_action[1]
        current_ward_nurses = self.ward_nurses_base + meta_action[2]

        # Simulate patient arrivals & discharges for 15 mins (stochastic mock)
        self.er_census += np.random.poisson(2)
        self.icu_census += np.random.poisson(0.5)
        self.ward_census += np.random.poisson(1)

        # Calculate capacities based on ratios
        max_er_capacity = current_er_nurses * self.er_ratio_limit
        max_icu_capacity = current_icu_nurses * self.icu_ratio_limit
        max_ward_capacity = current_ward_nurses * self.ward_ratio_limit

        # Basic safety constraint check for the CMDP filter[cite: 2]
        safety_violation = 0
        if self.er_census > max_er_capacity or self.icu_census > max_icu_capacity or self.ward_census > max_ward_capacity:
            safety_violation = 1 

        self.current_step += 1
        done = self.current_step >= (24 * 60) / self.time_step_mins # Done after 24 simulated hours

        # Base reward (throughput) - Nihara will integrate the equity penalty here[cite: 2]
        reward = -(self.er_census + self.icu_census + self.ward_census) 
        
        info = {
            "safety_violation": safety_violation,
            "er_ratio": self.er_census / (current_er_nurses + 1e-5),
            "icu_ratio": self.icu_census / (current_icu_nurses + 1e-5),
            "ward_ratio": self.ward_census / (current_ward_nurses + 1e-5)
        }

        return self._get_obs(), reward, done, info

if __name__ == "__main__":
    env = HospitalEnv()
    obs = env.reset()
    print(f"Initial State Observation: {obs}")
    
    # Test a dummy action (allocating 3 float nurses to ER, 1 to ICU, 2 to Ward)
    action = [3, 1, 2]
    next_obs, reward, done, info = env.step(action)
    print(f"Action Taken: {action}")
    print(f"Next State Observation: {next_obs}")
    print(f"Safety Violation Triggered: {bool(info['safety_violation'])}")
    print("Simulation engine test passed!")