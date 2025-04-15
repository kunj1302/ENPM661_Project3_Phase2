# ENPM661_Project3_Phase2
"""
A* Path Planning Implementation for Falcon Sim Turtlebot Controller
====================================================================

This module implements second methods of A* path planning that have been experimented
with in our Falcon Sim project. First methodd is generating A* path and is later processed and passed
to the Turtlebot controller for simulation in Falcon Sim.

--------------------------------------------------------------------------------
Method 2: Direct In-Script A* Implementation
--------------------------------------------------------------------------------
Description:
    - The A* algorithm is implemented directly within the `astar_planner.py` script.
    - The algorithm continuously runs within the same script and generates the path on the fly.
    - After generation, the resulting path data is processed (including coordinate flips,
      scaling to match the Falcon Sim canvas, and calculation of motion deltas) and passed
      to the Turtlebot controller.
    
    
Cons:
    - When used as part of the Falcon Sim planner (with multiple function calls across the script),
      the controller behavior does not match the expected 2D motion perfectly.
    - The Turtlebot motion exhibits slight misbehavior during simulation - likely due to
      function interaction and timing issues within the integrated Falcon Sim environment.
    - The same A* algorithm works as expected in a standalone Python test but loses consistency
      when integrated across several functions in Falcon Sim.
      
--------------------------------------------------------------------------------
Usage Instructions:
    1. Ensure that the ROS2 launch file passes the start and end positions, robot parameters,
       and motion parameters correctly.
    2. For Method 1:
          - Pre-calculate the path based on the obstacle map.
          - The generated path (formatted as a list of [x, y, theta] waypoints) is then processed
            in the `path_plan` function to adjust coordinates to match the canvas.
          - Use the processed path to drive the Turtlebot via the controller.
    3. For Method 2:
          - The integrated A* algorithm in this script directly generates the path.
          - Post-process the generated path (e.g., compute deltas) and supply it to the controller.
          - Note that current integration issues may cause slight misbehavior in simulation.
            Further investigation on function interaction and timing is required.
    4. Adjust the grid generation (obstacle map) and coordinate transformations as needed
       to ensure consistent unit usage across the system (e.g. all in centimeters or meters).
       
--------------------------------------------------------------------------------
Notes:
    - Both methods have been useful for learning and prototyping, though Method 1 has proven
      more robust within Falcon Sim.
    - Future improvements may involve:
          a) Enhancing dynamic replanning.
          b) Refining integration between the planning and controller modules.
          c) Achieving smoother coordination of multiple functions in Falcon Sim for
             consistent Turtlebot motion.

"""
