#!/usr/bin/env python3
"""
PD controller for MySinglePegInsertion environment.
This script implements a basic PD controller to run the MySinglePegInsertion task
without RL, by guiding the arms to pick up the objects and perform insertion.
"""

import time
import numpy as np
import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
import os

# Force CPU execution for JAX
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
# os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Print JAX device info
print("JAX devices:", jax.devices())
print("Default JAX device:", jax.default_backend())

# Import the environment
from mujoco_playground._src.manipulation.aloha import my_single_peg_insertion
from mujoco_playground._src import mjx_env
import mediapy as media

class PDController:
    """
    Proportional-Derivative controller for the peg insertion task.
    This controller defines waypoints for both grippers to follow, guiding them to:
    1. Move to grasp positions
    2. Close grippers to grasp objects
    3. Lift objects up
    4. Move to pre-insertion position
    5. Perform insertion
    """
    def __init__(self, env):
        # Save reference to environment
        self.env = env
        
        # PD gains
        self.Kp = 5.0  # Proportional gain
        self.Kd = 0.5  # Derivative gain
        
        # Define waypoints for the task
        # Each waypoint is [left_gripper_pos(3), right_gripper_pos(3)]
        self.waypoints = [
            # Initial positions - approach objects
            {"left_pos": np.array([-0.05, 0.0, 0.1]), "right_pos": np.array([0.05, 0.0, 0.1]), 
             "left_grip": 0.0, "right_grip": 0.0, "duration": 25},
            
            # Move down to grasp objects
            {"left_pos": np.array([-0.05, 0.0, 0.03]), "right_pos": np.array([0.05, 0.0, 0.03]), 
             "left_grip": 0.0, "right_grip": 0.0, "duration": 25},
            
            # Close grippers
            {"left_pos": np.array([-0.05, 0.0, 0.03]), "right_pos": np.array([0.05, 0.0, 0.03]), 
             "left_grip": 1.0, "right_grip": 1.0, "duration": 15},
            
            # Lift up
            {"left_pos": np.array([-0.05, 0.0, 0.15]), "right_pos": np.array([0.05, 0.0, 0.15]), 
             "left_grip": 1.0, "right_grip": 1.0, "duration": 25},
            
            # Move to pre-insertion position (socket on left, peg on right)
            {"left_pos": np.array([-0.05, 0.0, 0.15]), "right_pos": np.array([0.0, 0.0, 0.15]), 
             "left_grip": 1.0, "right_grip": 1.0, "duration": 25},
            
            # Perform insertion
            {"left_pos": np.array([-0.01, 0.0, 0.1]), "right_pos": np.array([0.0, 0.0, 0.1]), 
             "left_grip": 1.0, "right_grip": 1.0, "duration": 35},
        ]
        
        # Current waypoint index
        self.current_waypoint = 0
        self.steps_at_waypoint = 0
        
    def compute_action(self, state):
        """Compute PD control action based on current state and target waypoint"""
        # Extract current positions from state
        data = state.data
        left_gripper_pos = data.site_xpos[self.env._left_gripper_site]
        right_gripper_pos = data.site_xpos[self.env._right_gripper_site]
        
        # Get the current waypoint
        if self.current_waypoint >= len(self.waypoints):
            # If we've completed all waypoints, just stay in place
            waypoint = self.waypoints[-1]
        else:
            waypoint = self.waypoints[self.current_waypoint]
            self.steps_at_waypoint += 1
            
            # Check if we should move to the next waypoint
            if self.steps_at_waypoint >= waypoint["duration"]:
                self.current_waypoint += 1
                self.steps_at_waypoint = 0
                if self.current_waypoint < len(self.waypoints):
                    waypoint = self.waypoints[self.current_waypoint]
        
        # Compute position errors
        left_pos_error = waypoint["left_pos"] - left_gripper_pos
        right_pos_error = waypoint["right_pos"] - right_gripper_pos
        
        # Simple PD control (no velocity for now, just proportional)
        left_action = self.Kp * left_pos_error
        right_action = self.Kp * right_pos_error
        
        # Set gripper actions
        left_grip_action = waypoint["left_grip"]  # Scalar value
        right_grip_action = waypoint["right_grip"]  # Scalar value
        
        # Get the current action dimension
        action_dim = self.env.action_size
        
        # Combine all actions (for ALOHA, typically 14 dims: 6 DOF per arm + 1 for each gripper)
        action = np.zeros(action_dim)  
        
        # Apply to each arm's first 3 joints (simplified)
        # For ALOHA: typically joints 0-5 are left arm, 7-12 are right arm, 6 and 13 are grippers
        # This is a simplification - adjust indices based on actual robot configuration
        action[0:3] = left_action * 0.01  # Scale down to reasonable joint delta
        action[7:10] = right_action * 0.01  
        
        # Apply to grippers
        action[6] = left_grip_action
        action[13] = right_grip_action
        
        return action

def run_simulation():
    """Run the simulation with a PD controller"""
    init_start_time = time.time()
    
    # Create the environment
    config = my_single_peg_insertion.default_config()
    env = my_single_peg_insertion.MySinglePegInsertion(config=config)
    
    print(f"Environment action dimension: {env.action_size}")
    
    # Create the PD controller
    controller = PDController(env)
    
    # Reset the environment
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    
    print(f"Control shape: {state.data.ctrl.shape}")
    
    init_end_time = time.time()
    print(f"Initialization took {init_end_time - init_start_time:.2f} seconds")
    
    # Initialize trajectory list to store frames
    # Only store key frames instead of every frame - reduce to fewer frames
    key_frames_indices = [0, 50, 100, 149]  # Start, middle, near end, end
    trajectory = [] # Store only the selected key frames
    
    # Run simulation for a reduced number of steps
    n_steps = 150  # Reduced from 400
    print(f"Starting simulation loop with {n_steps} steps...")
    sim_loop_start_time = time.time()
    
    # Add initial state if it's a keyframe
    if 0 in key_frames_indices:
         trajectory.append(state)
            
    # JIT compile the step function for potential speedup on CPU too
    # Note: This compiles based on the *shapes* of the initial state and action
    # If shapes change later (unlikely here), it would recompile.
    jitted_step = jax.jit(env.step)
    
    # Get initial action for compiling
    initial_action = controller.compute_action(state)
    initial_action_jax = jax.numpy.array(initial_action[:state.data.ctrl.shape[0]])
    print("Compiling env.step with JIT...")
    compile_start_time = time.time()
    # Run the JIT compilation step
    state = jitted_step(state, initial_action_jax)
    # Force completion of the compilation step and associated computations
    state.data.qpos.block_until_ready()
    compile_end_time = time.time()
    print(f"JIT compilation took {compile_end_time - compile_start_time:.2f} seconds.")
    
    # Add the state after the first (compilation) step if it's a keyframe
    if 1 in key_frames_indices: # Check if step 1 (index 0 in loop) is needed
        trajectory.append(state)
        print(f"Step 1 completed, storing frame.")
        
    for i in range(1, n_steps): # Start loop from 1 since we did step 0 for compilation
        step_start_time = time.time()
        
        # Compute control action using PD controller
        action = controller.compute_action(state)
        
        # Convert to jax array with the right shape
        action_jax = jax.numpy.array(action[:state.data.ctrl.shape[0]])
        
        # Ensure previous computations affecting the action are done (more robust timing)
        action_jax.block_until_ready()
        
        # Step the environment using the JIT-compiled function
        state = jitted_step(state, action_jax)
        
        # Ensure step completes before timing by blocking on an output array
        state.data.qpos.block_until_ready() 
        
        step_end_time = time.time()
        # Print timing for the first few steps to gauge step time
        if i < 5:
            print(f"  Step {i} took {step_end_time - step_start_time:.4f} seconds")
            
        # Only store state if it's a key frame index
        if (i + 1) in key_frames_indices: # i+1 because loop starts from 1
            trajectory.append(state)
            print(f"Step {i+1} completed, storing frame.")
    
    sim_loop_end_time = time.time()
    print(f"Simulation loop completed in {sim_loop_end_time - sim_loop_start_time:.2f} seconds")
    print(f"Saved {len(trajectory)} frames out of {n_steps} steps")
    
    # Render trajectory
    print("Rendering trajectory...")
    render_start_time = time.time()
    
    fps = 2.0  # Even lower FPS since we have very few frames
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    
    # Use render_array function from mjx_env
    frames = mjx_env.render_array(
        env.mj_model,
        trajectory, # Now contains only key frames
        height=240, # Reduced resolution
        width=320,  # Reduced resolution
        scene_option=scene_option
    )
    
    render_end_time = time.time()
    print(f"Rendering {len(frames)} frames took {render_end_time - render_start_time:.2f} seconds")
    
    # Display video
    media.show_video(frames, fps=fps)
    
    # Save video to file
    video_path = "peg_insertion_pd_sparse_cpu.mp4"
    media.write_video(video_path, frames, fps=fps)
    print(f"Video saved to {video_path}")
    
    return frames

if __name__ == "__main__":
    print("Starting PD controller simulation on CPU...")
    total_start_time = time.time()
    run_simulation()
    total_end_time = time.time()
    print(f"Total script execution time: {total_end_time - total_start_time:.2f} seconds")
    print("Simulation complete!") 