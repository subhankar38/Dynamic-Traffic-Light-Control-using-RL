
import os
import sys
import optparse
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from config import (
    SUMO_HOME, HYPERPARAMS, MODEL_DIR, PLOT_DIR,
    get_sumo_config_path, get_tripinfo_output_path, DEFAULT_SUMO_SCENARIO
)
from traffic_env import TrafficEnv
from agent import Agent
from model import Model

# Ensure SUMO_HOME is set
if SUMO_HOME in os.environ:
    tools = os.path.join(os.environ[SUMO_HOME], "tools")
    sys.path.append(tools)
else:
    sys.exit(f"Please declare environment variable '{SUMO_HOME}'")
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="traffic_model",
        help="Name of the model file to save/load",
    )
    optParser.add_option(
        "--train",
        action='store_true',
        default=False,
        help="Enable training mode (default: testing mode)",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=HYPERPARAMS['epochs'],
        help="Number of epochs for training",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=HYPERPARAMS['steps_per_epoch'],
        help="Number of simulation steps per epoch",
    )
    optParser.add_option(
        "--scenario",
        dest='scenario_name',
        type='string',
        default=DEFAULT_SUMO_SCENARIO, # Use default from config
        help="Name of the SUMO scenario folder (e.g., 'city1', 'citysample')",
    )
    options, args = optParser.parse_args()
    return options

def run(train_mode=True, model_name="traffic_model", epochs=HYPERPARAMS['epochs'], 
            steps_per_epoch=HYPERPARAMS['steps_per_epoch'], scenario_name=DEFAULT_SUMO_SCENARIO):
        sumo_cfg_path = get_sumo_config_path(scenario_name)
        tripinfo_output_path = get_tripinfo_output_path(scenario_name)

        if not os.path.exists(sumo_cfg_path):
            sys.exit(f"Error: SUMO configuration file not found for scenario '{scenario_name}' at {sumo_cfg_path}")


        # Setup TensorBoard writer for logging metrics
        log_dir = os.path.join(PLOT_DIR, "tensorboard_logs", model_name, time.strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")

        best_total_waiting_time = np.inf
        total_waiting_time_history = []

        # Initialize the TrafficEnv with dynamic paths
        traffic_env = TrafficEnv(sumo_cfg=sumo_cfg_path, tripinfo_output=tripinfo_output_path)
        all_junction_ids = traffic_env.get_traffic_light_ids() # This will temporarily start/stop SUMO
        
        # Map junction IDs to numerical indices for the agent's memory
        junction_map = {id: i for i, id in enumerate(all_junction_ids)}
        junction_numbers = list(junction_map.values()) # Numerical indices for junctions

        # Initialize the Agent
        brain = Agent(
            gamma=HYPERPARAMS['gamma'],
            epsilon=HYPERPARAMS['epsilon_start'] if train_mode else HYPERPARAMS['epsilon_end'],
            lr=HYPERPARAMS['learning_rate'],
            input_dims=HYPERPARAMS['input_dims'],
            fc1_dims=HYPERPARAMS['fc1_dims'],
            fc2_dims=HYPERPARAMS['fc2_dims'],
            batch_size=HYPERPARAMS['batch_size'],
            n_actions=HYPERPARAMS['n_actions'],
            junctions=junction_numbers,
            max_memory_size=HYPERPARAMS['max_memory_size'],
            epsilon_dec=HYPERPARAMS['epsilon_decay'],
            epsilon_end=HYPERPARAMS['epsilon_end'],
        )

        # Load model if in test mode or continuing training
        model_path = os.path.join(MODEL_DIR, f"{model_name}.bin")
        if not train_mode and os.path.exists(model_path):
            brain.Q_eval.load_state_dict(torch.load(model_path, map_location=brain.Q_eval.device))
            print(f"Loaded model from {model_path}")
        elif train_mode and os.path.exists(model_path):
            print(f"Model exists at {model_path}, continuing training from there.")
            brain.Q_eval.load_state_dict(torch.load(model_path, map_location=brain.Q_eval.device))
        else:
            print("Starting training with a new model (or model not found for test mode).")


        print(f"Using device: {brain.Q_eval.device}")

        # Main training/testing loop
        for epoch in range(epochs):
            print(f"\n--- Epoch: {epoch + 1}/{epochs} ---")
            
            # Start SUMO simulation for the current epoch
            traffic_env.start_sumo(gui=not train_mode)
            
            current_epoch_total_waiting_time = 0
            min_phase_duration = HYPERPARAMS['min_phase_duration']

            # Reset agent's memory counters for new epoch if needed
            brain.reset(junction_numbers)

            # Initialize traffic light specific variables
            traffic_lights_phase_time_left = {id: 0 for id in all_junction_ids}
            prev_vehicles_per_lane = {junction_map[id]: [0] * HYPERPARAMS['input_dims'] for id in all_junction_ids}
            prev_action = {junction_map[id]: 0 for id in all_junction_ids} # Initial action for each junction

            for step in range(steps_per_epoch):
                # Advance SUMO simulation by one step
                traffic_env.simulation_step()

                for junction_id in all_junction_ids:
                    junction_num = junction_map[junction_id]
                    controlled_lanes = traffic_env.get_controlled_lanes(junction_id)
                    current_waiting_time = traffic_env.get_total_waiting_time_for_lanes(controlled_lanes)
                    current_epoch_total_waiting_time += current_waiting_time

                    # If the current phase for this traffic light has ended
                    if traffic_lights_phase_time_left[junction_id] == 0:
                        vehicles_per_lane = traffic_env.get_vehicle_numbers_on_lanes(controlled_lanes)
                        
                        # Store transition (state, action, reward, next_state, done)
                        reward = -1 * current_waiting_time # Negative reward for waiting time
                        current_state_obs = list(vehicles_per_lane.values()) 
                        
                        # Ensure state dimension matches input_dims (pad with zeros if necessary)
                        padded_current_state_obs = current_state_obs + [0] * (HYPERPARAMS['input_dims'] - len(current_state_obs))
                        padded_prev_state_obs = prev_vehicles_per_lane[junction_num] # This already has the correct padded size if set from previous step

                        brain.store_transition(
                            padded_prev_state_obs,
                            padded_current_state_obs,
                            prev_action[junction_num],
                            reward,
                            (step == steps_per_epoch - 1), # 'done' if it's the last step
                            junction_num
                        )

                        prev_vehicles_per_lane[junction_num] = padded_current_state_obs # Store padded state for next step

                        # Choose a new action (traffic light phase)
                        action = brain.choose_action(padded_current_state_obs)
                        prev_action[junction_num] = action

                        # Apply the new traffic light phase
                        traffic_env.set_traffic_light_phase(
                            junction_id,
                            HYPERPARAMS['min_phase_duration'],
                            HYPERPARAMS['traffic_light_phases'][action]
                        )

                        # Reset phase timer
                        traffic_lights_phase_time_left[junction_id] = min_phase_duration + HYPERPARAMS['green_phase_duration_offset']
                        
                        # Learn from experiences if in training mode and enough memory
                        if train_mode:
                            loss = brain.learn(junction_num)
                            if loss is not None:
                                writer.add_scalar(f'Agent/Loss/Junction_{junction_num}', loss, epoch * steps_per_epoch + step)
                    else:
                        traffic_lights_phase_time_left[junction_id] -= 1
            
            # End of epoch: log total waiting time
            print(f"Epoch {epoch + 1}: Total Waiting Time = {current_epoch_total_waiting_time}")
            total_waiting_time_history.append(current_epoch_total_waiting_time)
            writer.add_scalar('Epoch/Total_Waiting_Time', current_epoch_total_waiting_time, epoch)
            writer.add_scalar('Agent/Epsilon', brain.epsilon, epoch)

            # Save the model if it's the best performing
            if current_epoch_total_waiting_time < best_total_waiting_time:
                best_total_waiting_time = current_epoch_total_waiting_time
                if train_mode:
                    brain.save_model(model_name)
                    print(f"Model saved to {model_path} (new best waiting time: {best_total_waiting_time})")

            # Close SUMO simulation for this epoch
            traffic_env.close_sumo()
            sys.stdout.flush()

            if not train_mode:
                break # Exit after one epoch if in test mode

        # Final plot of total waiting time history (if training)
        if train_mode:
            # Using matplotlib to save plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(list(range(len(total_waiting_time_history))), total_waiting_time_history)
            plt.xlabel("Epochs")
            plt.ylabel("Total Waiting Time (SUMO steps)")
            plt.title("Total Waiting Time vs. Epochs")
            plt.grid(True)
            plt.savefig(os.path.join(PLOT_DIR, f'time_vs_epoch_{model_name}_{scenario_name}.png'))
            # plt.show() # Disabled for headless environments

        writer.close()
        print("Training/Testing completed. TensorBoard writer closed.")


    # Main entry point when the script is run
if __name__ == "__main__":
    options = get_options()
    run(
            train_mode=options.train,
            model_name=options.model_name,
            epochs=options.epochs,
            steps_per_epoch=options.steps,
            scenario_name=options.scenario_name # Pass the chosen scenario
    )
    