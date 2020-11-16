# Proposed plan

## From references:
 - Most RL research in space/orbit trajectory optimization 
   found so far focuses on low-thrust spacecraft and/or 
   interplanetary trajectory optimisation, but there is 
   also research in orbital prediction, planetary landing 
   and spacecraft guidance.
 - A common theme seems to be that the purpose of the RL agents 
   is to find an initial guesses for a genetic optimization 
   algorithm.


## Thesis goals
### Research goal
Research the feasibility of using a trained RL agent
to generate (initial) approximations for ground to orbit 
trajectories of a winged low earth orbit launcher. The goal is
to minimize the time for computing an optimal trajectory.
 - Can an RL agent be trained to solve this problem?
 - How would such an agent perform compared to existing 
   methods?
 - Can such an RL agent be use in conjunction with one or
   more existing methods to accelerate convergence of the 
   optimal trajectory estimate.
   
### Develop
 * RL based method
 * Generic algorithm based method
 * Simple/analytical method (gravity turn, etc)

### Compare
 * All methods individually.
 * Initial RL run with further optimization using the genetic algorithm.
 * Initial simple method run with further optimization using the genetic 
   algorithm.


## Preliminary thesis
The goal of the preliminary thesis is to collect literature and experiment
with simple systems in order to adjust the overall scope of the main thesis.

### Assumptions
 * 2D simulation of on a circular celestial body
    - Assumes the target orbital plane is aligned with the
      launch heading.
 * A lunar surface-to-orbit launcher
    - Lunar SSTO's exist. Examples are the Apollo Lunar Module 
      and several Luna program spacecraft. So there is reference 
      material for the basic design of a single stage lunar 
      launcher.
    - Smaller than the earth → shorter simulations → faster training → more experiments.
    - No atmosphere.
 * Basic launcher
    - No wings, just a conventional launcher
    
### Experiments
 1. **Single stage suborbital launcher.**
    - Purpose
       - Verify development setup.
    - Agent goal: Reach the highest altitude.
    - Reward based on
      * Altitude
    - State space
      * Engine state
      * Propellant state
    - Actions
      * Ignite engine (Multiple ignitions not possible)
      * Cuttoff engine
    - Expected optimal behaviour
      1. Ignite engine at simulation start
      2. Do not cuttoff the engine until after
         all propellant has burned.

 2. **Two stage suborbital launcher.**
    - Purpose
       - Verify development setup.
    - Agent goal: Reach the highest altitude.
    - Reward based on
      * Altitude
    - State space
      * Engine state
      * Propellant state
    - Actions
      * Ignite engine (Multiple ignitions of the same engine not possible)
      * Cuttoff engine
      * Drop stage
    - Expected optimal behaviour
      1. Ignite first stage engine at simulation start
      2. Do not cuttoff the engine until after
         all propellant has burned.
      4. Drop first stage close after burnout
      5. Ignite second stage.
      6. Do not cuttoff the engine until after
         all propellant has burned.
          
 3. **Single stage to orbit launcher.**
    - Purpose
       - Research options in RL methods.
       - Check for limits in what's possible within the timeframe of the
         thesis.
    - Agent goal: Launch from the ground to a target orbit
    - Reward based on
      * Error in orbital elements (eg, eccentricity, semimajor axis)
    - State space
      * Position
      * Velocity
      * Engine state
      * Propellant state
      * (Target) orbital elements
    - Actions
      * Ignite engine (Multiple ignitions possible, but negative reward 
        each time?)
      * Cuttoff engine
      * Flight path angle or pitch angle or pitch rate.
    - Expected optimal behaviour
      1. Ignite engine at simulation start
      2. Cuttoff engine when an appropriate apogee has been reached
      3. Restart engine to circularize orbit/adjust eccentricity.
          
 4. **Two stage to orbit launcher.**
    - Purpose
       - Research options in RL methods.
       - Check for limits in what's possible within the timeframe of the
         thesis.
    - Agent goal: Launch from the ground to a target orbit
    - Reward
      * Error in orbital elements (eccentricity, semimajor axis)
    - State space
      * Position
      * Velocity
      * Engine state
      * Propellant state
      * (Target) orbital elements
    - Actions
      * Ignite engine (Multiple ignitions of the first stage not possible)
      * Cuttoff engine
      * Flight path angle or pitch angle or pitch rate.
    - Expected optimal behaviour
      1. Ignite first stage engine at simulation start.
      2. Do not cuttoff the engine until after
         all propellant has burned.
      4. Drop first stage close after burnout.
      5. Ignite second stage.
      6. Cuttoff engine when an appropriate apogee has been reached.
      7. Restart engine to circularize orbit/adjust eccentricity.
 
 5. **Terraform the moon and add an atmosphere**


## Thesis experiments
The goal would be to get rid of some of the assumptions made during
the preliminary theses and add features. So the plan would be to add 
the following features one by one. The exact details will depend on
the results from the preliminary thesis.
 1. Earth ground-to-orbit launcher.
 2. Add design parameters to the agent state space. (Eg, mass, aerodynamics)
 3. Wings aka lift.
 4. 3D simulation (JSBSim). Orbital plane not aligned with launch heading.
 5. Wind
 6. Extra constraints
     - First stage glide-back check. Give negative rewards if it's
       not possible for the first stage to glide back to the landing
       site.
     - Restricted airspace.
     - Mid flight failure. Adjust trajectory to prevent parts falling 
       outside of a defined safety area in case of structural failure
       or FTS activation.
 7. Fun extra, run computed trajectories in Kerbal Space Program (with real 
    solar system mod).
