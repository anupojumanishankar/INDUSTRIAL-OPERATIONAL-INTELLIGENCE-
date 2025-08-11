import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import math

def simulate_60kw_motor_data(start_time, days=10):
    """
    Simulate 10 days of 60kW industrial motor data with realistic electrical behavior
    Including load variations, faults, noise, and typical motor characteristics
    """
    
    # Motor nameplate specifications
    RATED_POWER = 60  # kW
    RATED_VOLTAGE = 415  # V (line-to-line)
    RATED_CURRENT = 107  # A at full load
    BASE_PF = 0.85  # Power factor at full load
    BASE_EFFICIENCY = 0.92  # Efficiency at full load
    
    num_entries = days * 288  # 288 entries per day (5 min intervals)
    data = []
    
    # Initialize cumulative energy counters
    total_kwh = 0
    total_kvah = 0
    total_kvarh = 0
    
    # Motor state tracking
    motor_running = True
    trip_remaining = 0
    power_off_remaining = 0
    starting_sequence = 0
    
    for i in range(num_entries):
        row = {}
        
        # Basic identifiers
        timestamp = start_time + timedelta(minutes=5 * i)
        row['TS'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        row['DID'] = "E_AA_Z_A_Z_P0040_D1"
        row['FID'] = "V1.0"
        row['SLAVE_ID'] = 114
        row['RSSI'] = round(random.uniform(-45, -85), 1)
        
        # Random events and motor states
        random_event = random.random()
        
        # Power off events (1-3 hours randomly)
        if random_event < 0.001 and power_off_remaining == 0:  # 0.1% chance
            power_off_remaining = random.randint(12, 36)  # 1-3 hours in 5-min intervals
        
        # Motor trips (random equipment faults)
        if random_event < 0.002 and trip_remaining == 0 and power_off_remaining == 0:  # 0.2% chance
            trip_remaining = random.randint(6, 24)  # 30min to 2 hours
        
        # Motor starting sequence
        if motor_running == False and power_off_remaining == 0 and trip_remaining == 0:
            if random_event < 0.1:  # 10% chance to start when not running
                starting_sequence = 3  # 15 minutes starting sequence
                motor_running = True
        
        # Handle power off state
        if power_off_remaining > 0:
            power_off_remaining -= 1
            motor_running = False
            # All values are zero or null during power off
            for key in ['KWH', 'KVAH', 'KVARH', 'VR', 'VY', 'VB', 'AVG_VLN', 
                       'VRY', 'VYB', 'VBR', 'AVG_VLL', 'IR', 'IY', 'IB', 'AVG_I',
                       'R_PF', 'Y_PF', 'B_PF', 'AVG_PF', 'FREQUENCY', 'R_KW', 'Y_KW', 
                       'B_KW', 'TOTAL_KW', 'R_KVA', 'Y_KVA', 'B_KVA', 'TOTAL_KVA',
                       'R_KVAR', 'Y_KVAR', 'B_KVAR', 'TOTAL_KVAR', 'R_THD-V', 
                       'Y_THD-V', 'B_THD-V', 'R_THD-I', 'Y_THD-I', 'B_THD-I']:
                row[key] = None if random.random() < 0.3 else 0  # 30% chance of null
            data.append(row)
            continue
        
        # Handle trip state
        if trip_remaining > 0:
            trip_remaining -= 1
            motor_running = False
            # Voltage present but no current/power
            row['VR'] = round(random.uniform(405, 425) * random.uniform(0.95, 1.05), 2)
            row['VY'] = round(random.uniform(405, 425) * random.uniform(0.95, 1.05), 2)
            row['VB'] = round(random.uniform(405, 425) * random.uniform(0.95, 1.05), 2)
            row['AVG_VLN'] = round((row['VR'] + row['VY'] + row['VB']) / 3 / math.sqrt(3), 2)
            
            row['VRY'] = round(math.sqrt(3) * random.uniform(405, 425), 2)
            row['VYB'] = round(math.sqrt(3) * random.uniform(405, 425), 2)
            row['VBR'] = round(math.sqrt(3) * random.uniform(405, 425), 2)
            row['AVG_VLL'] = round((row['VRY'] + row['VYB'] + row['VBR']) / 3, 2)
            
            row['FREQUENCY'] = round(random.uniform(49.8, 50.2), 2)
            
            # Zero currents and power during trip
            for key in ['IR', 'IY', 'IB', 'AVG_I', 'R_KW', 'Y_KW', 'B_KW', 'TOTAL_KW',
                       'R_KVA', 'Y_KVA', 'B_KVA', 'TOTAL_KVA', 'R_KVAR', 'Y_KVAR', 
                       'B_KVAR', 'TOTAL_KVAR', 'R_PF', 'Y_PF', 'B_PF', 'AVG_PF']:
                row[key] = 0
            
            # Cumulative energies remain same
            row['KWH'] = round(total_kwh, 2)
            row['KVAH'] = round(total_kvah, 2)
            row['KVARH'] = round(total_kvarh, 2)
            
            # THD values
            row['R_THD-V'] = round(random.uniform(1.0, 3.0), 2)
            row['Y_THD-V'] = round(random.uniform(1.0, 3.0), 2)
            row['B_THD-V'] = round(random.uniform(1.0, 3.0), 2)
            row['R_THD-I'] = 0
            row['Y_THD-I'] = 0
            row['B_THD-I'] = 0
            
            data.append(row)
            continue
        
        # Normal operation or starting sequence
        motor_running = True
        
        # Determine load percentage (40% to 100% with realistic variations)
        if starting_sequence > 0:
            # Starting current surge
            starting_sequence -= 1
            load_percent = random.uniform(150, 600)  # Starting current surge
            current_multiplier = load_percent / 100
        else:
            # Normal load variations based on typical industrial patterns
            hour = timestamp.hour
            if 6 <= hour <= 18:  # Working hours - higher load
                base_load = random.uniform(60, 95)
            elif 19 <= hour <= 22:  # Evening - medium load
                base_load = random.uniform(45, 75)
            else:  # Night/early morning - lower load
                base_load = random.uniform(40, 65)
            
            # Add random variations and occasional overload
            load_variation = random.uniform(-15, 25)
            if random.random() < 0.05:  # 5% chance of overload
                load_variation += random.uniform(10, 30)
            
            load_percent = max(40, min(120, base_load + load_variation))
            current_multiplier = load_percent / 100
        
        # Calculate power factor and efficiency based on load
        if load_percent >= 100:
            pf_base = BASE_PF * random.uniform(0.95, 1.02)
            efficiency = BASE_EFFICIENCY * random.uniform(0.98, 1.01)
        elif load_percent >= 75:
            pf_base = 0.82 * random.uniform(0.95, 1.05)
            efficiency = 0.915 * random.uniform(0.98, 1.02)
        elif load_percent >= 50:
            pf_base = 0.75 * random.uniform(0.92, 1.08)
            efficiency = 0.90 * random.uniform(0.96, 1.04)
        else:
            pf_base = 0.60 * random.uniform(0.85, 1.15)
            efficiency = 0.85 * random.uniform(0.92, 1.08)
        
        # Voltage with imbalance and variations
        voltage_base = random.uniform(405, 425)  # Line-to-line voltage variation
        voltage_imbalance = random.uniform(0.5, 3.5)  # % imbalance
        
        vry = voltage_base * (1 + random.uniform(-voltage_imbalance/100, voltage_imbalance/100))
        vyb = voltage_base * (1 + random.uniform(-voltage_imbalance/100, voltage_imbalance/100))
        vbr = voltage_base * (1 + random.uniform(-voltage_imbalance/100, voltage_imbalance/100))
        
        # Line to neutral voltages
        vr = vry / math.sqrt(3) * random.uniform(0.98, 1.02)
        vy = vyb / math.sqrt(3) * random.uniform(0.98, 1.02)
        vb = vbr / math.sqrt(3) * random.uniform(0.98, 1.02)
        
        row['VR'] = round(vr, 2)
        row['VY'] = round(vy, 2)
        row['VB'] = round(vb, 2)
        row['AVG_VLN'] = round((vr + vy + vb) / 3, 2)
        row['VRY'] = round(vry, 2)
        row['VYB'] = round(vyb, 2)
        row['VBR'] = round(vbr, 2)
        row['AVG_VLL'] = round((vry + vyb + vbr) / 3, 2)
        
        # Current calculations with imbalance
        current_base = RATED_CURRENT * current_multiplier
        current_imbalance = random.uniform(1.0, 8.0)  # % imbalance
        
        ir = current_base * (1 + random.uniform(-current_imbalance/100, current_imbalance/100))
        iy = current_base * (1 + random.uniform(-current_imbalance/100, current_imbalance/100))
        ib = current_base * (1 + random.uniform(-current_imbalance/100, current_imbalance/100))
        
        row['IR'] = round(ir, 2)
        row['IY'] = round(iy, 2)
        row['IB'] = round(ib, 2)
        row['AVG_I'] = round((ir + iy + ib) / 3, 2)
        
        # Power factor per phase with variations
        pf_variation = random.uniform(0.95, 1.05)
        r_pf = max(0.1, min(1.0, pf_base * pf_variation))
        y_pf = max(0.1, min(1.0, pf_base * random.uniform(0.96, 1.04)))
        b_pf = max(0.1, min(1.0, pf_base * random.uniform(0.97, 1.03)))
        
        row['R_PF'] = round(r_pf, 3)
        row['Y_PF'] = round(y_pf, 3)
        row['B_PF'] = round(b_pf, 3)
        row['AVG_PF'] = round((r_pf + y_pf + b_pf) / 3, 3)
        
        # Frequency with minor variations
        row['FREQUENCY'] = round(random.uniform(49.7, 50.3), 2)
        
        # Power calculations with losses and imbalances
        total_kw_ideal = RATED_POWER * (load_percent / 100)
        
        # Add motor losses
        motor_losses = total_kw_ideal * (1 - efficiency) / efficiency
        total_kw_actual = total_kw_ideal + motor_losses
        
        # Distribute power among phases (with imbalance)
        r_kw = total_kw_actual / 3 * random.uniform(0.85, 1.15)
        y_kw = total_kw_actual / 3 * random.uniform(0.90, 1.10)
        b_kw = total_kw_actual / 3 * random.uniform(0.88, 1.12)
        
        row['R_KW'] = round(r_kw, 2)
        row['Y_KW'] = round(y_kw, 2)
        row['B_KW'] = round(b_kw, 2)
        row['TOTAL_KW'] = round(r_kw + y_kw + b_kw, 2)
        
        # KVA calculations
        r_kva = r_kw / r_pf if r_pf > 0.1 else r_kw / 0.1
        y_kva = y_kw / y_pf if y_pf > 0.1 else y_kw / 0.1
        b_kva = b_kw / b_pf if b_pf > 0.1 else b_kw / 0.1
        
        row['R_KVA'] = round(r_kva, 2)
        row['Y_KVA'] = round(y_kva, 2)
        row['B_KVA'] = round(b_kva, 2)
        row['TOTAL_KVA'] = round(r_kva + y_kva + b_kva, 2)
        
        # KVAR calculations
        r_kvar = math.sqrt(max(0, r_kva**2 - r_kw**2))
        y_kvar = math.sqrt(max(0, y_kva**2 - y_kw**2))
        b_kvar = math.sqrt(max(0, b_kva**2 - b_kw**2))
        
        row['R_KVAR'] = round(r_kvar, 2)
        row['Y_KVAR'] = round(y_kvar, 2)
        row['B_KVAR'] = round(b_kvar, 2)
        row['TOTAL_KVAR'] = round(r_kvar + y_kvar + b_kvar, 2)
        
        # THD calculations based on load and motor condition
        base_thd_v = 1.5 + (load_percent - 40) * 0.02  # Higher load = slightly higher THD-V
        base_thd_i = 3.0 + (load_percent - 40) * 0.08  # Higher load = higher THD-I
        
        if starting_sequence > 0:
            base_thd_i *= 2.5  # Higher THD during starting
        
        row['R_THD-V'] = round(base_thd_v * random.uniform(0.7, 1.8), 2)
        row['Y_THD-V'] = round(base_thd_v * random.uniform(0.8, 1.6), 2)
        row['B_THD-V'] = round(base_thd_v * random.uniform(0.75, 1.7), 2)
        row['R_THD-I'] = round(base_thd_i * random.uniform(0.6, 1.4), 2)
        row['Y_THD-I'] = round(base_thd_i * random.uniform(0.7, 1.3), 2)
        row['B_THD-I'] = round(base_thd_i * random.uniform(0.65, 1.35), 2)
        
        # Cumulative energy calculations (only when motor is running)
        energy_increment = (row['TOTAL_KW'] * 5/60) / 1000  # kWh in 5 minutes
        kva_increment = (row['TOTAL_KVA'] * 5/60) / 1000   # kVAh in 5 minutes  
        kvar_increment = (row['TOTAL_KVAR'] * 5/60) / 1000 # kVARh in 5 minutes
        
        total_kwh += energy_increment
        total_kvah += kva_increment
        total_kvarh += kvar_increment
        
        row['KWH'] = round(total_kwh, 3)
        row['KVAH'] = round(total_kvah, 3)
        row['KVARH'] = round(total_kvarh, 3)
        
        # Add random noise and null values (MFM meter issues)
        noise_probability = random.random()
        if noise_probability < 0.05:  # 5% chance of data corruption
            # Randomly corrupt some values
            corrupt_fields = random.sample([
                'VR', 'VY', 'VB', 'IR', 'IY', 'IB', 'R_KW', 'Y_KW', 'B_KW',
                'R_PF', 'Y_PF', 'B_PF', 'FREQUENCY'
            ], random.randint(1, 4))
            
            for field in corrupt_fields:
                if random.random() < 0.3:  # 30% chance of null
                    row[field] = None
                else:  # 70% chance of noisy data
                    if field in ['VR', 'VY', 'VB']:
                        row[field] = row[field] * random.uniform(0.5, 1.8)  # Voltage spikes/dips
                    elif field in ['IR', 'IY', 'IB']:
                        row[field] = row[field] * random.uniform(0.3, 2.5)  # Current spikes
                    elif 'PF' in field:
                        row[field] = max(0, min(1, row[field] * random.uniform(0.7, 1.3)))
                    elif field == 'FREQUENCY':
                        row[field] = row[field] * random.uniform(0.95, 1.05)
                    else:
                        row[field] = row[field] * random.uniform(0.6, 1.7)
        
        data.append(row)
    
    return pd.DataFrame(data)

# Generate 10 days of data starting from current date minus 10 days
current_time = datetime.now()
start_time = current_time - timedelta(days=10)

print("Generating 60kW industrial motor simulation data...")
print(f"Start time: {start_time}")
print(f"Generating 10 days of data with 5-minute intervals...")

# Generate the dataset
df_motor = simulate_60kw_motor_data(start_time, days=10)

print(f"Dataset generated successfully!")
print(f"Total records: {len(df_motor)}")
print(f"Date range: {df_motor['TS'].min()} to {df_motor['TS'].max()}")
print(f"Columns: {len(df_motor.columns)}")

# Display basic statistics
print("\nDataset Overview:")
print(f"Power range: {df_motor['TOTAL_KW'].min():.2f} - {df_motor['TOTAL_KW'].max():.2f} kW")
print(f"Current range: {df_motor['AVG_I'].min():.2f} - {df_motor['AVG_I'].max():.2f} A")
print(f"Voltage range: {df_motor['AVG_VLL'].min():.2f} - {df_motor['AVG_VLL'].max():.2f} V")
print(f"Power Factor range: {df_motor['AVG_PF'].min():.3f} - {df_motor['AVG_PF'].max():.3f}")

# Save to CSV
output_filename = "60kW_Motor_Simulation_Data.csv"
df_motor.to_csv(output_filename, index=False)
print(f"\nData saved to: {output_filename}")