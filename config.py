
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_HOME = "SUMO_HOME" 
SUMO_DATA_DIR = os.path.join(BASE_DIR, "sumo_data")
DEFAULT_SUMO_SCENARIO = "city1"
TRIPINFO_OUTPUT_FILENAME = "tripinfo.xml"

MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
    
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "tensorboard_logs"), exist_ok=True)


HYPERPARAMS = {
    'gamma': 0.99,
    'epsilon_start': 0.0,          
    'epsilon_end': 0.05,           
    'epsilon_decay': 5e-4,         
    'learning_rate': 0.1,          
    'input_dims': 4,               
    'fc1_dims': 256,               
    'fc2_dims': 256,               
    'n_actions': 4,                
    'batch_size': 1024,            
    'max_memory_size': 100000,     
    
    
    'epochs': 100,                  
    'steps_per_epoch': 1000,        
    'min_phase_duration': 5,
    'green_phase_duration_offset': 10,
}
HYPERPARAMS['traffic_light_phases'] = [
        ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],  # Phase 0
        ["rrryyyrrrrrr", "rrrGGGrrrrrr"],  # Phase 1
        ["rrrrrryyyrrr", "rrrrrrGGGrrr"],  # Phase 2
        ["rrrrrrrrryyy", "rrrrrrrrrGGG"],  # Phase 3
]

def get_sumo_scenario_path(scenario_name=DEFAULT_SUMO_SCENARIO):
    return os.path.join(SUMO_DATA_DIR, scenario_name)

def get_sumo_config_path(scenario_name=DEFAULT_SUMO_SCENARIO):
        
    scenario_folder = get_sumo_scenario_path(scenario_name)
    return os.path.join(scenario_folder, f"{scenario_name}.sumocfg")

def get_tripinfo_output_path(scenario_name=DEFAULT_SUMO_SCENARIO):
    scenario_folder = get_sumo_scenario_path(scenario_name)
    return os.path.join(scenario_folder, TRIPINFO_OUTPUT_FILENAME)
    