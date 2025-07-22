# traffic_env.py
import os
import sys

# Ensure SUMO_HOME is set to import sumolib and traci
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary # noqa
import traci # noqa

class TrafficEnv:
    def __init__(self, sumo_cfg, tripinfo_output="tripinfo.xml"):
        self.sumo_cfg = sumo_cfg
        self.tripinfo_output = tripinfo_output
        self.sumo_cmd = None
        self.traffic_light_ids = None

    def start_sumo(self, gui=False):
        sumo_binary = checkBinary("sumo-gui") if gui else checkBinary("sumo")
        self.sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--tripinfo-output", self.tripinfo_output
        ]
        traci.start(self.sumo_cmd)
        self.traffic_light_ids = traci.trafficlight.getIDList()
        print(f"SUMO simulation started with GUI: {gui}")

    def close_sumo(self):
        traci.close()

    def simulation_step(self):
        traci.simulationStep()

    def get_traffic_light_ids(self):
        if self.traffic_light_ids is None:
            # If not initialized, connect temporarily to get IDs
            self.start_sumo(gui=False)
            ids = traci.trafficlight.getIDList()
            self.close_sumo()
            self.traffic_light_ids = ids
        return self.traffic_light_ids

    def get_controlled_lanes(self, traffic_light_id):
        return traci.trafficlight.getControlledLanes(traffic_light_id)

    def get_vehicle_numbers_on_lanes(self, lanes):
        vehicle_per_lane = dict()
        for lane_id in lanes:
            vehicle_per_lane[lane_id] = 0
            # Consider vehicles that are not too close to the intersection
            for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id):
                if traci.vehicle.getLanePosition(vehicle_id) > 5.0: # Position threshold
                    vehicle_per_lane[lane_id] += 1
        return vehicle_per_lane

    def get_total_waiting_time_for_lanes(self, lanes):
        waiting_time = 0.0
        for lane_id in lanes:
            waiting_time += traci.lane.getWaitingTime(lane_id)
        return waiting_time

    def set_traffic_light_phase(self, junction_id, duration_yellow, phase_states):
        yellow_state = phase_states[0]
        green_state = phase_states[1]

        # Apply yellow phase first
        traci.trafficlight.setRedYellowGreenState(junction_id, yellow_state)
        traci.trafficlight.setPhaseDuration(junction_id, duration_yellow)
  
        # Then apply green phase (this is handled by SUMO's phase definition)
        # We set the duration for the whole phase, including implicit yellow if defined in .tll.xml
        # If the phases are simple, directly setting the green phase with total duration
        # (yellow + green) is common for single-step control.
        # This simplifies the control logic by having SUMO handle the actual phase transitions
        # based on the duration we provide.
        traci.trafficlight.setRedYellowGreenState(junction_id, green_state)
        # The phase duration is set for the combined effective "green" time
        # The agent determines the total duration including the yellow/transition.
        # The min_phase_duration + green_phase_duration_offset is the total duration set
        # by the agent for the current chosen green phase.
        
