# Original author: Tawn Kramer
# Completly remade and repurposed by: Javier Moralejo

"""Example of automatic vehicle control from client side."""

import asyncore
import base64
import collections
import datetime
import glob
import math
import os
import random
import re
import signal
import sys
import time
import weakref
from io import BytesIO
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from config import (
    ACTION_NOISE,
    BASE_REWARD,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CRASH_SPEED_WEIGHT,
    INPUT_DIM,
    LEADING_INSTRUCTIONS,
    MAX_CTE_ERROR,
    MAX_STEERING,
    MAX_THROTTLE,
    MIN_STEERING,
    MIN_THROTTLE,
    REWARD_CRASH,
    ROI,
    THROTTLE_REWARD_WEIGHT,
)

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_SPACE, KMOD_CTRL, K_q, K_r
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    sys.path.append(
        glob.glob(
            "PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append("PythonAPI/carla")
except IndexError:
    pass

import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from carla import ColorConverter as cc

ROAD_OPTIONS = [
    RoadOption.STRAIGHT,
    RoadOption.LEFT,
    RoadOption.RIGHT,
]  # , RoadOption.LANEFOLLOW, RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]

DEFAULT = ["Default"]
CLEAR = ["ClearNoon", "ClearSunset"]
CLOUD = ["CloudyNoon", "CloudySunset"]
WET = ["WetNoon", "WetSunset"]
CLOUDWET = ["WetCloudyNoon", "WetCloudySunset"]
STORM = [
    "HardRainNoon",
    "HardRainSunset",
    "MidRainSunset",
    "MidRainyNoon",
    "SoftRainNoon",
    "SoftRainSunset",
]
MAPS = ["Town01", "Town03", "Town04"]


def signal_handler(sig):
    print("ByeBye")
    sys.exit(0)


class CarlaSimContoller:
    """
    Wrapper for communicating with unity simulation.

    :param level: (int) Level index
    :param port: (int) Port to use for communicating with the simulator
    :param max_cte_error: (float) Max cross track error before reset
    """

    def __init__(self, level, port=2000, max_cte_error=3.0, vae=None):
        self.level = level
        self.verbose = False

        self.address = ("localhost", port)

        pygame.init()
        pygame.font.init()

        self.loaded = False

        self.min_throttle = MIN_THROTTLE
        self.max_throttle = MAX_THROTTLE

        self.world_frame = 0
        self.current_img = np.zeros(INPUT_DIM)
        self.current_sem = np.zeros(INPUT_DIM)

        self.speed = 0
        self.cte = 0
        self.angle_diff = 0
        self.instruction = None
        self.success = 0

        self.last_throttle = 0
        self.last_steering = 0

        self.collision_history = None
        self.lane_crossed = False

        self.episode = 0
        self.map_num = 0

        self.speed_array = [10] * 40

        try:
            self.client = carla.Client("localhost", port)
            self.client.set_timeout(100.0)
            self.display = pygame.display.set_mode(
                (CAMERA_WIDTH, CAMERA_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            self.hud = HUD(CAMERA_WIDTH, CAMERA_HEIGHT)
            self.world = World(self.client.get_world(), self.hud, vae=vae)
            self.controller = KeyboardControl(self.world)
            self.clock = pygame.time.Clock()
            self.loaded = True
        except:
            print("Error 1")

    def close_connection(self):
        return True

    def wait_until_loaded(self):
        """
        Wait for a client (Unity simulator).
        """
        while not self.loaded:
            print(
                "Waiting for sim to start..."
                "if the simulation is running, press EXIT to go back to the menu"
            )
            time.sleep(3.0)

    def reset(self, next_weather=None):
        self.world_frame = 0
        self.current_img = np.zeros(INPUT_DIM)
        self.current_sem = np.zeros(INPUT_DIM)

        self.speed = 0
        self.cte = 0
        self.angle_diff = 0
        self.instruction = None
        self.is_junction = 0
        self.success = 0

        self.last_throttle = 0
        self.last_steering = 0

        self.collision_history = None
        self.lane_crossed = False

        self.speed_array = [10] * 40

        self.world.restart(next_weather=next_weather)
        self.controller.restart()

    def get_sensor_size(self):
        """
        :return: (int, int, int)
        """
        return

    def take_action(self, action):
        self.clock.tick_busy_loop(60)
        if self.controller.parse_events():
            print("Bye Bye")
            self.quit()

        self.world.world.wait_for_tick(100.0)
        self.world.tick(self.clock)
        self.world.render(self.display)
        pygame.display.flip()

        control = carla.VehicleControl()

        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        self.last_throttle = action[1]
        self.last_steering = action[0]

        control.steer = float(action[0])

        if action[1] >= 0:
            control.throttle = float(action[1])
        else:
            control.brake = float(-action[1])
        self.world.player.apply_control(control)

    def observe(self):
        """
        :return: (np.ndarray)
        """
        self.world_frame = self.world.hud.frame
        self.collision_history = self.world.collision_sensor.history
        self.lane_crossed = self.world.lane_invasion_sensor.lane_crossed
        self.cte, self.angle_diff, self.instruction, self.is_junction, self.success = (
            self.world.get_ct_angle_dif_instruction()
        )
        car_rotation = self.world.player.get_transform().rotation
        r = Rotation.from_euler(
            "XZY", [car_rotation.roll, car_rotation.yaw, car_rotation.pitch], degrees=True
        )
        accel = self.world.player.get_acceleration()
        angular_vel = self.world.player.get_angular_velocity()
        angular_vel = [angular_vel.x, angular_vel.y, angular_vel.z]
        vel = self.world.player.get_velocity()

        road_opt = []

        for i in range(LEADING_INSTRUCTIONS):
            instruction_array = [0] * len(ROAD_OPTIONS)
            if (
                self.instruction[i] != None
                and self.instruction[i] != RoadOption.VOID
                and self.instruction[i] != RoadOption.LANEFOLLOW
            ):
                instruction_array[ROAD_OPTIONS.index(self.instruction[i])] = 1
            road_opt += instruction_array
        if (
            self.instruction[0] == RoadOption.CHANGELANELEFT
            or self.instruction[0] == RoadOption.CHANGELANERIGHT
        ):
            print(self.instruction[0])

        self.current_img = self.world.camera_manager.rgb_image
        self.current_sem = self.world.semantic_manager.semantic_image

        self.speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        self.speed_array[1 : len(self.speed_array)] = self.speed_array[
            0 : len(self.speed_array) - 1
        ]
        self.speed_array[0] = self.speed

        vel = r.apply([vel.x, -vel.y, vel.z])
        accel = r.apply([accel.x, -accel.y, accel.z])

        info = {
            "speed": self.speed,
            "cte": self.cte,
            "angle_diff": self.angle_diff,
            "throttle": self.last_throttle,
            "steering": self.last_steering,
            "vel": vel,
            "accel": accel,
            "angular_vel": angular_vel,
            "road_opt": road_opt,
            "is_junction": self.is_junction,
            "success": self.success,
        }

        return self.world.camera_manager.rgb_image, self.calc_reward(), self.is_game_over(), info

    def quit(self):
        if self.world is not None:
            self.world.destroy()

            pygame.quit()

    def render(self, mode):
        pass

    def is_game_over(self):
        if self.collision_history:
            print("Collision")
            return True

        if self.controller.crash:
            print("key crash")
            return True

        if all([s <= 2 for s in self.speed_array]):
            print("Too Slow")
            return True

        if abs(self.cte) > MAX_CTE_ERROR + self.is_junction:
            print("CTE")
            return True

        return False

    def calc_reward(self):
        if self.collision_history:

            # if self.world_frame - self.collision_history[-1][0] < 10:
            # print("Crash: ", REWARD_CRASH - self.collision_history[-1][1] * CRASH_SPEED_WEIGHT)
            # return REWARD_CRASH - self.collision_history[-1][1] * CRASH_SPEED_WEIGHT
            return REWARD_CRASH - self.speed * CRASH_SPEED_WEIGHT
        if self.controller.crash:
            # print("reward crash")
            return REWARD_CRASH - self.speed * CRASH_SPEED_WEIGHT

        # if self.lane_crossed:
        #    return REWARD_CRASH - self.speed * CRASH_SPEED_WEIGHT
        if all([s <= 2 for s in self.speed_array]):
            print("Too Slow")
            return 0

        if abs(self.cte) > MAX_CTE_ERROR:

            transform = self.world.player.get_transform()

            location = transform.location

            cte = 9999999
            close_w = None

            for w, _ in self.world.route:
                d = w.transform.location.distance(location)

                if d < cte:
                    close_w = w
                cte = min(cte, d)

            print("CAR: ", transform)
            print("WP: ", close_w.transform)
            print("Instruction: ", self.instruction)

            return REWARD_CRASH - self.speed * CRASH_SPEED_WEIGHT

        # return BASE_REWARD + THROTTLE_REWARD_WEIGHT * self.last_throttle

        return self.last_throttle + (1 - abs(self.cte) / MAX_CTE_ERROR) + (1 - abs(self.angle_diff))

        return 0


class World(object):
    """Class representing the surrounding environment"""

    def __init__(self, carla_world, hud, vae=None):
        """Constructor method"""
        self.vae = vae

        self.world = carla_world

        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print("RuntimeError: {}".format(error))
            print("  The server could not send the OpenDRIVE (.xodr) file:")
            print("  Make sure it exists, has the same name of your town, and is correct.")
            sys.exit(1)

        for a in self.world.get_actors():
            a.set_simulate_physics(enabled=False)

        self.seed = 43  # 42
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.semantic_manager = None
        self.route = None
        self._grp = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = "vehicle.tesla.model3"
        self._gamma = 2.2
        self.restart()
        self.world.on_tick(self.hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, next_weather=None):
        """Restart the world"""

        # self.next_weather(next_weather = next_weather)

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Set the seed if requested by user
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
        self.seed += 1

        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute("role_name", "hero")
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        # Spawn the player.
        print("Spawning the player")
        if self.player is not None:
            # spawn_point = self.player.get_transform()
            # spawn_point.location.z += 2.0
            # spawn_point.rotation.roll = 0.0
            # spawn_point.rotation.pitch = 0.0
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

            if self.player is not None:
                i = 0
                while (
                    self.player.get_location().x == 0
                    and self.player.get_location().y == 0
                    and self.player.get_location().z == 0
                ):
                    i += 1
                    if i == 50000:
                        print("stuck5")
                    pass
                destination_point = (
                    random.choice(spawn_points) if spawn_points else carla.Transform()
                )
                i = 0

                while spawn_point == destination_point:
                    i += 1
                    if i == 50000:
                        print("stuck6")
                    destination_point = (
                        random.choice(spawn_points) if spawn_points else carla.Transform()
                    )
                self.route = self.set_destination(destination_point.location)

                if (
                    self.route[0][0].transform.location.distance(self.player.get_location())
                    >= MAX_CTE_ERROR
                ):
                    self.player.set_transform(self.route[0][0].transform)

                idx = 0
                while (
                    self.route[idx][1] == RoadOption.CHANGELANELEFT
                    or self.route[idx][1] == RoadOption.CHANGELANERIGHT
                ) and (idx < len(self.route)):
                    idx += 1
                self.player.set_transform(self.route[idx][0].transform)

        j = 0
        while self.player is None:
            j += 1
            if j > 50000:
                print("stuck7")
            if not self.map.get_spawn_points():
                print("There are no spawn points available in your map/town.")
                print("Please add some Vehicle Spawn Point to your UE4 scene.")
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

            if self.player is not None:
                i = 0
                while (
                    self.player.get_location().x == 0
                    and self.player.get_location().y == 0
                    and self.player.get_location().z == 0
                ):
                    i += 1
                    if i == 50000:
                        print("stuck3")
                    pass

                destination_point = (
                    random.choice(spawn_points) if spawn_points else carla.Transform()
                )
                i = 0
                while spawn_point == destination_point:
                    i += 1
                    if i == 50000:
                        print("stuck4")
                    destination_point = (
                        random.choice(spawn_points) if spawn_points else carla.Transform()
                    )
                self.route = self.set_destination(destination_point.location)

                if (
                    self.route[0][0].transform.location.distance(self.player.get_location())
                    >= MAX_CTE_ERROR
                ):
                    self.player.set_transform(self.route[0][0].transform)

                idx = 0
                while (
                    self.route[idx][1] == RoadOption.CHANGELANELEFT
                    or self.route[idx][1] == RoadOption.CHANGELANERIGHT
                ):
                    idx += 1
                self.player.set_transform(self.route[idx][0].transform)

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        control.manual_gear_shift = True
        control.gear = 1
        self.player.apply_control(control)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)

        self.camera_manager = CameraManager(self.player, self.hud, self._gamma, vae=self.vae)
        # self.camera_manager.transform_index = cam_pos_id
        # self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.set_sensor(0, notify=False)

        self.semantic_manager = CameraManager(self.player, self.hud, self._gamma)
        self.semantic_manager.set_sensor(4, notify=False)

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        i = 0
        while (
            all(0 == pix for pix in self.camera_manager.rgb_image[0, 0])
            and all(0 == pix for pix in self.camera_manager.rgb_image[0, -1])
            and all(0 == pix for pix in self.camera_manager.rgb_image[-1, 0])
            and (0 == pix for pix in self.camera_manager.rgb_image[-1, -1])
        ):
            i += 1
            if i == 50000:
                print("stuck1")
            pass
        i = 0
        while (
            all(0 == pix for pix in self.semantic_manager.semantic_image[0, 0])
            and all(0 == pix for pix in self.semantic_manager.semantic_image[0, -1])
            and all(0 == pix for pix in self.semantic_manager.semantic_image[-1, 0])
            and (0 == pix for pix in self.semantic_manager.semantic_image[-1, -1])
        ):
            i += 1
            if i == 50000:
                print("stuck2")
            pass

    def next_weather(self, next_weather=None, reverse=False):
        """Get next weather setting"""
        """
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])
        """

        if next_weather is None:
            weather = ""
            n = random.random()
            if n < 0.1:
                weather = random.choice(DEFAULT)
            elif n < 0.37:
                weather = random.choice(CLEAR)
            elif n < 0.55:
                weather = random.choice(CLOUD)
            elif n < 0.68:
                weather = random.choice(WET)
            elif n < 0.84:
                weather = random.choice(CLOUDWET)
            else:
                weather = random.choice(STORM)
        else:
            weather = next_weather

        self.world.set_weather(getattr(carla.WeatherParameters, weather))

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.semantic_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player,
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """
        start_waypoint = self.map.get_waypoint(self.player.get_location())
        end_waypoint = self.map.get_waypoint(
            # carla.Location(location[0], location[1], location[2]))
            location
        )
        route_trace = self._trace_route(start_waypoint, end_waypoint)
        return route_trace

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self.map, 1.0)  # 0.5 #resolution
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location, end_waypoint.transform.location
        )

        return route

    def get_ct_angle_dif_instruction(self):
        transform = self.player.get_transform()

        location = transform.location
        rotation = transform.rotation.yaw

        distance = 9999999
        close_w = None
        close_ins = None

        for idx, (w, ins) in enumerate(self.route):
            d = w.transform.location.distance(location)

            if d < distance:
                close_w = w
                close_idx = idx
            distance = min(distance, d)

        w_rotation = close_w.transform.rotation.yaw
        w_location = close_w.transform.location

        angle_diff = rotation - w_rotation

        relative_position = (location.x - w_location.x, location.y - w_location.y)
        distance = math.sqrt(relative_position[0] ** 2 + relative_position[1] ** 2)

        relative_angle_rad = math.atan2(relative_position[1], relative_position[0])

        relative_angle_diff_rad = relative_angle_rad - math.radians(w_rotation)

        cte = math.sin(relative_angle_diff_rad) * distance

        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        angle_diff /= 180

        close_ins = []
        for i in range(LEADING_INSTRUCTIONS):
            if close_idx < len(self.route):
                close_ins.append(self.route[close_idx][1])
            else:
                close_ins.append(None)

        is_junction = int(close_w.is_junction)

        success = 0
        if close_idx == len(self.route) - 1:
            success = 1

        """
        print("\n")
        print("Trans: ", transform)
        print("W_Trans: ", close_w.transform)
        print("relative_position: ", relative_position)
        print("distance: ", distance)
        print("relative_angle: ", math.degrees(relative_angle_rad))
        print("relative_angle_diff: ", math.degrees(relative_angle_diff_rad))
        print("cte: ", cte)
        """

        return cte, angle_diff, close_ins, is_junction, success


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification("Collision with %r" % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.lane_crossed = False
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        self.lane_crossed = True
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.gnss")
        self.sensor = world.spawn_actor(
            blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent
        )
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

import time


def semantic_transformation(img):

    new_img = np.zeros((img.shape[0], img.shape[1], 3))
    for idx_x, x in enumerate(img):
        start = time.time()
        for idx_y, y in enumerate(x):
            if y[2] == 7:
                new_img[idx_x, idx_y, 0] = 1

            if y[2] == 6:
                new_img[idx_x, idx_y, 1] = 1

            if y[2] == 10 or y[2] == 13 or y[2] == 0:
                pass
            else:
                new_img[idx_x, idx_y, 2] = 1
    print(time.time() - start)
    return new_img


class CameraManager(object):
    """Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction, vae=None):
        """Constructor method"""
        self.vae = vae

        self.rgb_image = np.zeros(INPUT_DIM)
        self.semantic_image = np.zeros(INPUT_DIM)
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=0.4, z=1.3), carla.Rotation()), attachment.Rigid)
        ]

        self.transform_index = 1
        self.sensors = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB"],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)"],
            ["sensor.camera.depth", cc.Depth, "Camera Depth (Gray Scale)"],
            ["sensor.camera.depth", cc.LogarithmicDepth, "Camera Depth (Logarithmic Gray Scale)"],
            ["sensor.camera.semantic_segmentation", cc.Raw, "Camera Semantic Segmentation (Raw)"],
            [
                "sensor.camera.semantic_segmentation",
                cc.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)",
            ],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)"],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                blp.set_attribute("image_size_x", str(hud.dim[0]))
                blp.set_attribute("image_size_y", str(hud.dim[1]))
                if blp.has_attribute("gamma"):
                    blp.set_attribute("gamma", str(gamma_correction))
            elif item[0].startswith("sensor.lidar"):
                blp.set_attribute("range", "50")
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        self.transform_index = self.transform_index % len(self._camera_transforms)
        index = index % len(self.sensors)
        needs_respawn = (
            True
            if self.index is None
            else (force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        )
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: self._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            if self.sensors[self.index][2].startswith("Camera RGB"):
                self.rgb_image = cv2.resize(
                    array[ROI[1] : ROI[3], ROI[0] : ROI[2], :], (INPUT_DIM[1], INPUT_DIM[0])
                )
            elif self.sensors[self.index][2].startswith(
                "Camera Semantic Segmentation (CityScapes Palette)"
            ):
                self.semantic_image = cv2.resize(
                    array[ROI[1] : ROI[3], ROI[0] : ROI[2], :], (INPUT_DIM[1], INPUT_DIM[0])
                )
            elif self.sensors[self.index][2].startswith("Camera Semantic Segmentation (Raw)"):
                self.semantic_image = cv2.resize(
                    array[ROI[1] : ROI[3], ROI[0] : ROI[2], :], (INPUT_DIM[1], INPUT_DIM[0])
                )
                # self.semantic_image = semantic_transformation(self.semantic_image)
            array = array[:, :, ::-1]
            if self.sensors[self.index][2].startswith("Camera RGB"):
                if self.vae is not None:
                    encoded = self.vae.encode(self.rgb_image)
                    array = np.squeeze(self.vae.decode(encoded))
                    array = cv2.resize(array, (CAMERA_WIDTH, CAMERA_HEIGHT))
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording:
            if self.sensors[self.index][0].startswith("sensor.camera.rgb"):
                image.save_to_disk("_out/x/{:08d}".format(image.frame))
            elif self.sensors[self.index][0].startswith("sensor.camera.semantic_segmentation"):
                image.save_to_disk("_out/y/{:08d}".format(image.frame))


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        self.world = world
        self.crash = False
        self.record = False
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_record(event.key):
                    pass
                if self._is_quit_shortcut(event.key):
                    return True

            if event.type == pygame.KEYDOWN:
                if event.key == K_SPACE:
                    self.crash = True
                    # print(self.crash)
                    return False

    def restart(self):
        self.crash = False

    # @staticmethod
    def _is_quit_shortcut(self, key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    # @staticmethod
    def _is_record(self, key):
        """Shortcut for quitting"""
        if key == K_r:
            self.record = not self.record


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = "courier" if os.name == "nt" else "mono"
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == "nt" else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []

        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = "N" if abs(transform.rotation.yaw) < 89.5 else ""
        heading += "S" if abs(transform.rotation.yaw) > 90.5 else ""
        heading += "E" if 179.5 > transform.rotation.yaw > 0.5 else ""
        heading += "W" if -0.5 > transform.rotation.yaw > -179.5 else ""
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter("vehicle.*")

        self._info_text = [
            "Server:  % 16.0f FPS" % self.server_fps,
            "Client:  % 16.0f FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % get_actor_display_name(world.player, truncate=20),
            "Map:     % 20s" % world.map.name,
            "Simulation time: % 12s" % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            "Heading:% 16.0f\N{DEGREE SIGN} % 2s" % (transform.rotation.yaw, heading),
            "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (transform.location.x, transform.location.y)),
            "GNSS:% 24s" % ("(% 2.6f, % 3.6f)" % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            "Height:  % 18.0f m" % transform.location.z,
            "",
        ]
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ("Throttle:", control.throttle, 0.0, 1.0),
                ("Steer:", control.steer, -1.0, 1.0),
                ("Brake:", control.brake, 0.0, 1.0),
                ("Reverse:", control.reverse),
                ("Hand brake:", control.hand_brake),
                ("Manual:", control.manual_gear_shift),
                "Gear:        %s" % {-1: "R", 0: "N"}.get(control.gear, control.gear),
            ]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [("Speed:", control.speed, 0.0, 5.556), ("Jump:", control.jump)]
        self._info_text += [
            "",
            "Collision:",
            collision,
            "",
            "Number of vehicles: % 8d" % len(vehicles),
        ]

        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]

        def dist(l):
            return math.sqrt(
                (l.x - transform.location.x) ** 2
                + (l.y - transform.location.y) ** 2
                + (l.z - transform.location.z) ** 2
            )

        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append("% 4dm %s" % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text("Error: %s" % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6)
                            )
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """Class for fading text"""

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split("\n")

        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")

    def name(x):
        return " ".join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


def bgra2bgr(bgra, background=(255, 255, 255)):
    row, col, ch = bgra.shape

    # assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype="float32")

    a = 0

    if ch == 4:
        b, g, r, a = bgra[:, :, 0], bgra[:, :, 1], bgra[:, :, 2], bgra[:, :, 3]
    elif ch == 3:
        b, g, r = bgra[:, :, 0], bgra[:, :, 1], bgra[:, :, 2]
    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = b * a + (1.0 - a) * B
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = r * a + (1.0 - a) * R

    return np.asarray(rgb, dtype="uint8")
