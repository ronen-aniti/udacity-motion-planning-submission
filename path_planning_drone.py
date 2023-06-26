import csv
from enum import Enum, auto
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import msgpack
import numpy as np
import time
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.frame_utils import global_to_local
from udacidrone.messaging import MsgID

class States(Enum):
	MANUAL = auto()
	ARMING = auto()
	DISARMING = auto()
	PLANNING = auto()
	WAYPOINT = auto()
	HOVER = auto()

class PathPlanner(Drone):
	def __init__(self, connection):
		super().__init__(connection)
		self.target_position = np.array([0.0, 0.0, 0.0, 0.0])
		self.waypoints = []
		self.in_mission = True
		self.check_state = {}
		
		# Initial state
		self.flight_state = States.MANUAL

		# Register callbacks
		self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
		self.register_callback(MsgID.STATE, self.state_callback)

		# Deadband radius
		self.deadband_radius = 2.0

	def local_position_callback(self):
		# Current position in NEA frame
		current_position_nea = np.array([self.local_position[0], self.local_position[1], -self.local_position[2]])
		
		# Trigger a waypoint transition if the drone is within the deadband radius of the target position
		if self.flight_state == States.WAYPOINT:
			if len(self.waypoints) > 0:
				if np.linalg.norm(self.target_position[:3] - current_position_nea) < self.deadband_radius:
					self.waypoint_transition()
			else:
				self.hover_transition()

	def state_callback(self):
		if self.in_mission:
			if self.flight_state == States.MANUAL:
				self.arming_transition()
			elif self.flight_state == States.ARMING:
				if self.armed:
					self.plan_path()

	def plan_path(self):
		self.flight_state = States.PLANNING
		print("Planning a path using an RRT algorithm ...")

		# Get the home latitude and home longitude from the colliders file
		with open("colliders.csv") as file:
			reader = csv.reader(file)
			first_line = next(reader)
		home_lon = float(first_line[1].split(" lon0 ")[1])
		home_lat = float(first_line[0].split("lat0 ")[1])

		# Set the global home coordinates, assuming 0 meter altitude
		global_home = (home_lon, home_lat, 0.0)
		self.set_home_position(*global_home)

		# Get current global position
		global_position = self.global_position

		# Convert the global position to a NED position, then an NEA position
		current_local = global_to_local(global_position, global_home)
		current_local[2] = -current_local[2]
		
		print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
																				 self.local_position))

		# Import obstacles as array.
		obstacles = np.genfromtxt('colliders.csv', delimiter=',', skip_header=2)

		# Define a start position in the NEA frame
		start = current_local

		# Define a goal position in the NEA frame
		goal = np.array([475, -320 , 100]) #Other open grid positions to try: np.array([588, 456 , 5]) #np.array([588, 456 , 5])#np.array([475, -320 , 5])
		
		# Initialize the RRT planning object
		planner = RRT(start, obstacles)

		# Run the RRT algorithm to generate a 2D NumPy array of NEA positions
		path = planner.run(goal)

		# Convert the RRT output to a list of waypoints
		waypoints = [[p[0], p[1], p[2], 0] for p in path]

		# Set the waypoints attribute
		self.waypoints = waypoints 

		# Send waypoints to the simulator
		self.send_waypoints()
	
	def arming_transition(self):
		self.flight_state = States.ARMING
		print("Arming transition")
		self.arm()
		self.take_control()

	def waypoint_transition(self):
		self.flight_state = States.WAYPOINT
		print("Waypoint transition")
		self.target_position = self.waypoints.pop(0)
		print("Target position", self.target_position)
		self.cmd_position(*self.target_position)
	
	def hover_transition(self):
		self.flight_state = States.HOVER
		print("Hover transition")
	
	def send_waypoints(self):
		print("Sending waypoints to the simulator ...")
		data = msgpack.dumps(self.waypoints)
		self.connection._master.write(data)

	def start(self):
		self.start_log("Logs", "NavLog.txt")
		print("Starting connection")
		self.connection.start()
		self.stop_log()

class RRT:
	"""This is an implementation of the RRT algorithm. The algorithm is biased such that every 10th sample is the goal state."""
	def __init__(self, start, obstacles):
		"""Initializes the tree"""
		self.start = start
		self.goal = None
		self.explored = np.array([start])
		self.edges = {}		
		self.obstacle_centers = obstacles[:, :3]
		self.obstacle_halfsizes = obstacles[:, 3:]
		self.xbounds, self.ybounds, self.zbounds = self.bounds()
		self.lines = []

	def bounds(self):
		"""Determines the bounds of the obstacle map in the NEA frame"""
		obstacle_centers = self.obstacle_centers
		obstacle_halfsizes = self.obstacle_halfsizes 
		xmin = np.min(obstacle_centers[:,0] - obstacle_halfsizes[:,0])
		xmax = np.max(obstacle_centers[:,0] + obstacle_halfsizes[:,0])
		ymin = np.min(obstacle_centers[:,1] - obstacle_halfsizes[:,1])
		ymax = np.max(obstacle_centers[:,1] + obstacle_halfsizes[:,1])
		zmin = np.min(obstacle_centers[:,2] - obstacle_halfsizes[:,2])
		zmax = np.max(obstacle_centers[:,2] + obstacle_halfsizes[:,2])
		xbounds = (xmin, xmax)
		ybounds = (ymin, ymax)
		zbounds = (zmin, zmax)
		return xbounds, ybounds, zbounds

	def run(self, goal, step=15.0, max_iters=100000):
		"""Runs the main algorithm. This is a form of the rapidly-exploring random tree algorithm"""
		
		# Set the goal state
		self.goal = goal

		goal_found = False

		# Set the current state to be the start state
		current = self.start

		# Loop until the goal is reached
		iteration = 0
		while not self.goal_reached(current) and iteration < max_iters:
			iteration += 1
			s_sample = self.sample()
			if iteration % 10 == 0: 
				s_sample = self.goal

			# Returns the nearest tree node `s_nearest` to `s_sample`
			s_nearest_index = np.argmin(np.linalg.norm(s_sample[:3] - self.explored[:, :3], axis=1))
			s_nearest = self.explored[s_nearest_index]

			# Compute s_new, a state that is `step` distance away from s_nearest in the direction `u_hat` of s_sample
			u_hat = (s_sample - s_nearest) / np.linalg.norm(s_sample - s_nearest)
			s_new = s_nearest + u_hat * step
			
			# Add a branch of maximum length possible
			#for substep in np.linspace(step, 1.0, num=10):
			#	s_new = s_nearest + u_hat * substep
			#	if not self.collides(s_nearest, s_new):
			#		break

			# Add s_new to the tree only if the segment connecting it to s_nearest doesn't collide with any obstacles
			if not self.collides(s_nearest, s_new):
				self.explored = np.vstack((self.explored, s_new))
				# Add a key-value pair to `edges` to represent the connection between s_new and s_nearest
				s_new_index = len(self.explored) - 1
				self.edges[s_new_index] = s_nearest_index


			# Set the current node to be s_new
			current = s_new

			# Break the loop if the current node is close enough to the goal state
			if self.goal_reached(current):
				print("Goal is reached")
				goal_found = True
				break
				
		# Reconstruct and shorten the path if the goal is reached, then return it.
		if goal_found:
			path = self.reconstruct()
			# Appends the actual goal state to path if doing so doesn't register a collision 
			if not self.collides(current, self.goal):
				path = np.vstack((path, self.goal))
			path = self.shorten(path)
			print(path)
		
		# Call the animate method at the end of the run method
		#self.animate(path)
		
		return path

	def sample(self):
		"""Samples a random state from within the bounds of the obstacle space"""
		x = np.random.uniform(low=self.xbounds[0], high=self.xbounds[1], size=1)[0]
		y = np.random.uniform(low=self.ybounds[0], high=self.ybounds[1], size=1)[0]
		# Restrict z to goal altitude
		z = self.goal[2]
		#z = np.random.uniform(low=self.zbounds[0], high=self.zbounds[1], size=1)[0]
		return np.array([x, y, z])

	def collides(self, start_point, end_point):
		"""
		Checks if the bounding box of the line segment connecting the start point and the end point intersects the bounding 
		box of any obstacle from the obstacle space. This collision checking routine is known as the "Axis-aligned Bounding Box
		Algorithm" and serves the purpose in this code as a simple way of checking whether or not a sampled point should be added 
		to the tree. 
		"""
		obstacle_centers = self.obstacle_centers
		obstacle_halfsizes = self.obstacle_halfsizes
		# Defines the maxima and minima x, y, z positions of the bounding box around the line segment
		minima = np.minimum(start_point, end_point)
		maxima = np.maximum(start_point, end_point)

		# Checks whether or not the bounding box around each obstacle collides with the bounding box around the line segment. 
		collision_mask = np.all((obstacle_centers - obstacle_halfsizes) <= maxima, axis=1) & np.all((obstacle_centers + obstacle_halfsizes) >= minima, axis=1)

		# Returns True if a collision with any obstacle is detected and false otherwise
		return np.any(collision_mask)

	def goal_reached(self, current, tolerance=15.0):
		"""Checks whether or not the goal state has been reached within the tolerance specified"""
		current = np.array([current[0], current[1], current[2]])
		if np.linalg.norm(current - self.goal) < tolerance:
			return True
		return False


	def reconstruct(self):
		"""Reconstructs and returns the path from start state to goal state"""
		goal_index = len(self.explored) - 1
		start_index = 0
		path = [goal_index]
		current_index = goal_index
		while current_index != start_index:
			came_from_index = self.edges[current_index]
			path.append(came_from_index)
			current_index = came_from_index
		path = path[::-1]
		path = self.explored[path]
		return path

	def shorten(self, path):
		# Initialize an empty list for the shortened path
		shortened_path = [self.start]

		# Set the current state equal to the start state
		current_state = path[0]

		# Check for collision between the start state and all other states in the path
		collision_results = np.array([self.collides(path[0][:3], state[:3]) for state in path])

		# Get the maximum index that is False (indicating no collision)
		last_false_index = np.where(collision_results == False)[0][-1]

		# Append the path state corresponding to the max_false_index to the shortened_path list
		shortened_path.append(path[last_false_index])

		# Repeat steps 3-5 until reaching the end state
		while not np.array_equal(current_state, path[-1]):
			# Update the current state to be the last state added to the shortened path
			current_state = shortened_path[-1]

			# Check for collision between the current state and all other states in the path
			collision_results = np.array([self.collides(current_state[:3], state[:3]) for state in path])

			# Get the maximum index that is False (indicating no collision)
			last_false_index = np.where(collision_results == False)[0][-1]

			# Append the path state corresponding to the max_false_index to the shortened_path list
			if not np.array_equal(path[last_false_index], current_state):
				shortened_path.append(path[last_false_index])
		
		shortened_path = np.array(shortened_path)

		return shortened_path

if __name__ == "__main__":
	conn = MavlinkConnection('tcp:127.0.0.1:5760', threaded=False, PX4=False)
	drone = PathPlanner(conn)
	time.sleep(2)
	drone.start()