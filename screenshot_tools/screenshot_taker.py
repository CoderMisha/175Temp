from __future__ import print_function
from builtins import range
import MalmoPython
import os
import shutil
import sys
import time
import json
import numpy as np
from pynput.keyboard import Key, Controller

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

#fill in these parameters
screenshot_counter = 1 #number to start counting screenshot names at
current_user = "Misha" #Name attached to files so counters dont overlap
world_path = "" #path to world file, sometimes doesnt work and have to directly put in xml
screenshots_path = "" #path to screenshot folder
no_villagers_path = "" #path to folder of images without villagers
villagers_path = "" #path to folder of images with villagers
pixel_width = 512
pixel_height = 512
time_limit = 5000000
port_number = 10000

#you have to have mouse in window for screenshot
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
                            <min x="10" y="-10" z="0"/>
                            <max x="10" y="-10" z="0"/>
                        </Grid>
                    </ObservationFromGrid>                    
                    <ObservationFromFullStats/>
                    <ContinuousMovementCommands turnSpeedDegs="180"/>
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
agent_host.sendCommand("chat /kill @e[type=!player]")  
time.sleep(5)

# Loop until mission ends:
while world_state.is_mission_running:
    x = np.random.randint(200, 9999)
    if np.random.randint(0,2) > 0:
      x *= -1
    z = np.random.randint(200, 9999)
    if np.random.randint(0,2) > 0:
      z *= -1

    agent_host.sendCommand("chat /tp ~{} 90 ~{}".format(x,z))
    agent_host.sendCommand("chat /weather clear")
    time.sleep(1)
    agent_host.sendCommand("chat /kill @e[type=!player,r=200]")  
    time.sleep(1)
    agent_host.sendCommand("chat /kill @e[type=!player,r=200]")
    time.sleep(1)
    
    world_state = agent_host.getWorldState()
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
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

    if validLocation:
      keyboard.press(Key.f2)
      keyboard.release(Key.f2)
      time.sleep(0.5)

      villagerLocations = []
      for i in range(1,np.random.randint(4, 6)):
          x = np.random.beta(4, 3)*10
          z = np.random.normal(0, 2.5)
          agent_host.sendCommand("chat /summon villager ~{} ~5 ~{} {{Rotation:[90f,0f], Profession:1}}".format(x,z))
      time.sleep(2.5)
      
      keyboard.press(Key.f2)
      keyboard.release(Key.f2)
      time.sleep(0.5)

      for fileNum, filename in enumerate(os.listdir(screenshots_path)):
        src = screenshots_path + '\\' + filename
        if fileNum == 0:
          dst = no_villagers_path + '\\' + current_user + str(screenshot_counter) + "a.png"
        else:
          dst = villagers_path + '\\' + current_user + str(screenshot_counter) + "b.png"
        shutil.move(src, dst)
      screenshot_counter += 1
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

time.sleep(5)
print()
print("Mission ended")
# Mission has ended.
