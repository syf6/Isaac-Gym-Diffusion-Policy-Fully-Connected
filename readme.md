/home/shi/docker_isaac_gym_forenvironment/workspace/isaac-gym-ur3e/python/examples

v2 is the good example without force sensor on the destination,
v3 is the bad example with force sensor on the destination, i dont know if the force sensor is applied or not

v4 <-- v2 design a zone on the destination



state:
joint dof (robot arm position)

action: joystick action : Joystick event: type=2, code=0, value=14863    ---> this is not very proper for ddpm
or
action: joint dof (i think this is better for diffusion model ) 

problem grab : ?


End of 1 episode: 
