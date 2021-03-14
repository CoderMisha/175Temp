from __future__ import print_function
from builtins import range
import MalmoPython
import os
import shutil
import sys
import time
import json
import numpy as np
from PIL import Image
from pynput.keyboard import Key, Controller

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

#fill in these parameters
screenshot_counter = 1 #number to start counting screenshot names at
current_user = "Steve" #Name attached to files so counters dont overlap
world_path = "" #path to world file, sometimes doesnt work and have to directly put in xml
no_villagers_path = "" #path to folder of images without villagers
villagers_path = "" #path to folder of images with villagers
pixel_width = 256
pixel_height = 256
time_limit = 30000000 #time until auto end in milliseconds
port_number = 10000 #change to be able to run multiple minecrafts

#change villager type / ammount / location to your liking
#there are some checks for location

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Screenshot Taker</Summary>
              </About>
              
              <ServerSection>
               <ServerInitialConditions>
                  <Time>
                    <StartTime> 6000 </StartTime>
                    <AllowPassageOfTime> false </AllowPassageOfTime>
                  </Time>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FileWorldGenerator src= ''' + "\" " + world_path + "\"" + '''/>
                  <ServerQuitWhenAnyAgentFinishes/>
                  <ServerQuitFromTimeUp timeLimitMs=''' + "\' " + str(time_limit) + "\'" + '''/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Creative">
                <Name>ScreenShotBot</Name>
                <AgentStart>
                  <Placement x="0" y="100" z="0" yaw="-90"/>
                </AgentStart>
                <AgentHandlers>
                    <ChatCommands />
                    <ObservationFromGrid>
                        <Grid name="playerSpace">
                            <min x="0" y="-1" z="0"/>
                            <max x="0" y="1" z="0"/>
                        </Grid>
                        <Grid name="wallCheck">
                            <min x="0" y="1" z="-1"/>
                            <max x="3" y="1" z="1"/>
                        </Grid>
                        <Grid name="cliffCheck">
                            <min x="6" y="-3" z="0"/>
                            <max x="6" y="-3" z="0"/>
                        </Grid>
                    </ObservationFromGrid>                    
                    <ObservationFromFullStats/>
                    <VideoProducer>
                        <Width>''' + str(pixel_width) + '''</Width>
                        <Height>''' + str(pixel_height) + '''</Height>
                    </VideoProducer>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''


# Create default Malmo objects:

keyboard = Controller()
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", port_number))
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_client_pool ,my_mission_record, 0, "ScreenShotTaker" )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')
agent_host.sendCommand("chat /gamerule doMobSpawning false")
agent_host.sendCommand("chat /gamerule doMobLoot false")   
agent_host.sendCommand("chat /entitydata @e[type=!Player] {Health:0,DeathTime:19}") 
time.sleep(5)

# Loop until mission ends:
while world_state.is_mission_running:
    x = np.random.randint(200, 9999)
    if np.random.randint(0,2) > 0:
        x *= -1
    z = np.random.randint(200, 9999)
    if np.random.randint(0,2) > 0:
        z *= -1

    agent_host.sendCommand("chat /tp ~{} 100 ~{}".format(x,z))
    agent_host.sendCommand("chat /weather clear")
    time.sleep(2)
    agent_host.sendCommand("chat /entitydata @e[type=!Player] {Health:0,DeathTime:19}")  
    time.sleep(2)
    
    world_state = agent_host.getWorldState()
    try:
        msg = world_state.observations[-1].text
        observations = json.loads(msg)
    except:
        pass
        
    playerSpace = observations.get('playerSpace',['water','water','water'])
    wallCheck = observations.get('wallCheck',['water'])
    cliffCheck = observations.get('cliffCheck',['air'])
    validLocation = True

    for block in wallCheck:
        if block != 'air':
            validLocation = False
            break
    
    if cliffCheck[0] == 'air':
        validLocation = False

    if playerSpace[0] == 'leaves' or playerSpace[0] == 'air':
        validLocation = False

    if validLocation and len(world_state.video_frames) > 0:
        frame = world_state.video_frames[-1]
        im = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) )
        im.save(no_villagers_path + '\\' + current_user + str(screenshot_counter) + "a.png")
        time.sleep(0.5)

        villagerLocations = []
        for i in range(1,np.random.randint(3, 5)):
            x = np.random.beta(4, 3)*10
            z = np.random.normal(0, 2.5)
            agent_host.sendCommand("chat /summon villager ~{} ~5 ~{} {{Invulnerable:1, Rotation:[90f,0f], Profession:1}}".format(x,z))
        time.sleep(3)

        world_state = agent_host.getWorldState()
        if len(world_state.video_frames) > 0:
            frame = world_state.video_frames[-1]
            im = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) )
            im.save(villagers_path + '\\' + current_user + str(screenshot_counter) + "b.png")
            time.sleep(0.5)
            screenshot_counter += 1

    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

time.sleep(5)
print()
print("Mission ended")
# Mission has ended.
