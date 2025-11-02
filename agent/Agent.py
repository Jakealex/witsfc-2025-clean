from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        # self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation original
        self.init_pos = ([-14,0],[-8,-4],[-8,4],[-3,-3],[-4,3],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation new
        

    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance


    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    #--- OLD/ORIGINAL kickTarget()!!!--- #
    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""


    def select_skill(self, strategyData):
        """
        Main decision loop for each tick.
        Handles play modes, kickoff, and general gameplay decisions.
        """

        drawer = self.world.draw

        # ------------------------------------------------------
        # 🟢 Handle all game modes (new unified logic)
        mode_action = strategyData.get_mode_action()

        # === Phase 1: Handle reset and idle states ===
        if mode_action == "beam":
            return self.beam(True)  # re-beam players to starting formation

        elif mode_action == "idle":
            # IMPORTANT: always return a primitive action
            # Hold (or drift) toward our desired spot and face the ball during idle/set-play states
            return self.move(
                strategyData.my_desired_position,
                orientation=strategyData.ball_dir
            )

        # ------------------------------------------------------
        # 🟢 Phase 2: Handle our set plays (kickoffs, corners, goal kicks, etc.)
        if mode_action == "setplay":
            drawer.annotation((0, 10.5), "Our Set Play", drawer.Color.yellow, "status")

            # Use intelligent formations for the specific set play (corner, free kick, etc.)
            formation_positions = strategyData.get_setplay_positions()
            point_preferences = {
                i + 1: formation_positions[i] for i in range(len(formation_positions))
            }

            # If I'm the closest player to the ball, I take the restart
            if strategyData.player_unum == strategyData.get_closest_player_to_ball():
                pass_choice = strategyData.best_simple_pass_target()

                if pass_choice is not None:
                    _, pass_pos = pass_choice
                    drawer.annotation((0, 9.5), "Taking set play: pass", drawer.Color.cyan, "setplay_pass")
                    return self.kickTarget(strategyData, strategyData.mypos, pass_pos)
                else:
                    # If no nearby teammate, take a long kick toward goal
                    return self.kickTarget(strategyData, strategyData.mypos, (15, 0))

            # Otherwise, move into my assigned set-play position
            my_target = point_preferences[strategyData.player_unum]
            return self.move(my_target, orientation=strategyData.ball_dir)

        # ------------------------------------------------------
        # 🟠 Phase 3: Handle opponent set plays (we defend)
        if mode_action == "defend":
            drawer.annotation((0, 10.5), "Defending Set Play", drawer.Color.orange, "status")

            # Defensive formation near our box
            formation_positions = strategyData.get_setplay_positions()
            my_target = formation_positions[strategyData.player_unum - 1]

            return self.move(my_target, orientation=strategyData.ball_dir)

        # ------------------------------------------------------
        # 🟣 Phase 4: Handle Kickoffs (fallback if server mode still reports it)
        if strategyData.play_mode == strategyData.world.M_OUR_KICKOFF:
            drawer.annotation((0, 10.5), "Our Kickoff (legacy fallback)", drawer.Color.yellow, "status")

            formation_positions = strategyData.generate_dynamic_formation()
            point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)

            strategyData.my_desired_position = point_preferences[strategyData.player_unum]
            strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
                strategyData.my_desired_position
            )

            # Wait a bit before active player moves (simulates coordination)
            if strategyData.active_player_unum == strategyData.player_unum:
                return self.move(strategyData.ball_2d, orientation=strategyData.ball_dir)
            else:
                return self.move(strategyData.my_desired_position, orientation=strategyData.my_desired_orientation)

        elif strategyData.play_mode == strategyData.world.M_THEIR_KICKOFF:
            drawer.annotation((0, 10.5), "Their Kickoff (legacy fallback)", drawer.Color.yellow, "status")

            formation_positions = strategyData.generate_dynamic_formation()
            point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)

            strategyData.my_desired_position = point_preferences[strategyData.player_unum]
            strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
                strategyData.my_desired_position
            )

            return self.move(strategyData.my_desired_position, orientation=strategyData.my_desired_orientation)

        # ------------------------------------------------------
        # 🟢 Phase 5: Regular Play (PlayOn)
        # If none of the above conditions applied, we are in normal gameplay
        return self.strategy_decision(strategyData, None)


    # COMPLETELY NEW CODE #
    def strategy_decision(self, strategyData, point_preferences):
        """
        Handles play mode logic, formation-based positioning,
        and deciding when to move or kick.
        """

        drawer = self.world.draw

        # === 1. Generate dynamic formation and assign static roles ===
        formation_positions = strategyData.generate_dynamic_formation()
        point_preferences = {
            1: formation_positions[0],  # Goalkeeper
            2: formation_positions[1],  # LeftDefender
            3: formation_positions[2],  # RightDefender
            4: formation_positions[3],  # LeftForward
            5: formation_positions[4],  # RightForward
        }

        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )
        strategyData.my_role = strategyData.get_role_name_from_position(
            strategyData.my_desired_position, formation_positions
        )

        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")

        # --- 2. Handle non-PlayOn modes (Kickoff, Goal, etc.) ---
        if strategyData.play_mode != self.world.M_PLAY_ON:
            target = strategyData.handle_play_mode()
            drawer.annotation((0, 10.5), "Handling Play Mode", drawer.Color.yellow, "status")
            return self.move(target)

        # --- 3. If PlayOn: role-aware decisions ---
        closest_unum = strategyData.get_closest_player_to_ball()
        my_unum = strategyData.player_unum

        # --- 3a) Goalkeeper logic ---
        if getattr(strategyData, "my_role", None) == "Goalkeeper":
            ball_x, ball_y = strategyData.ball_2d
            mypos = strategyData.mypos
            distance_to_ball = np.linalg.norm(mypos - strategyData.ball_2d)

            # --- 1️⃣ Base GK positioning along goal line ---
            gk_x = -14.5  # fixed near goal line
            # Clamp lateral movement so GK doesn't drift wide
            gk_y = float(np.clip(ball_y, -2.5, 2.5))

            # Step slightly forward if ball is nearby (<6 m)
            if ball_x > -10:
                gk_x = -13.5
            if ball_x > -6:
                gk_x = -12.5

            gk_target = np.array([gk_x, gk_y])

            # --- 2️⃣ Emergency clearance if ball is dangerously close ---
            if distance_to_ball < 1.2:
                drawer.annotation((0, 10.5), "GK: Clearing ball!", drawer.Color.red, "gk_clear")
                target_goal = (15, 0)
                return self.kickTarget(strategyData, strategyData.mypos, target_goal)

            # --- 3️⃣ Normal guarding ---
            drawer.annotation((-13.5, 6.5), "GK: Guarding goal", drawer.Color.cyan, "gk_status")
            return self.move(gk_target, orientation=strategyData.ball_dir)

        # --- 3b) Defender logic ---
        if getattr(strategyData, "my_role", None) in ["LeftDefender", "RightDefender"]:
            ball_x, ball_y = strategyData.ball_2d
            my_role = strategyData.my_role
            my_unum = strategyData.player_unum
            closest_unum = strategyData.get_closest_player_to_ball()

            # === 1. EMERGENCY CLEAR ===
            if ball_x < -11 or my_unum == closest_unum:
                drawer.annotation((0, 10.5), f"{my_role}: Intercepting / Clearing!", drawer.Color.red, f"{my_role}_clear")
                target_goal = (15, 0)
                return self.kickTarget(strategyData, strategyData.mypos, target_goal)

            # === 2. ACTIVE DEFENDING ===
            if ball_x < 0:
                target_x = np.clip(-13 + (ball_x + 15) * 0.55, -13, -2)
                base_y = -3 if my_role == "LeftDefender" else 3
                target_y = float(np.clip(0.6 * ball_y + 0.4 * base_y, -6, 6))
                def_target = np.array([target_x, target_y])

                drawer.annotation(
                    (target_x, target_y + 1.5),
                    f"{my_role}: Tracking ball (our half)",
                    drawer.Color.green,
                    f"{my_role}_track",
                )
                return self.move(def_target, orientation=strategyData.ball_dir)

            # === 3. Ball in opponent half ===
            drawer.annotation((0, 10.5), f"{my_role}: Holding line", drawer.Color.green, f"{my_role}_hold")
            return self.move(strategyData.my_desired_position, orientation=strategyData.my_desired_orientation)

        # --- 3c) Forwards: spread wide and press upfield when attacking ---
        if getattr(strategyData, "my_role", None) in ["LeftForward", "RightForward"]:
            ball_x, ball_y = strategyData.ball_2d
            my_role = strategyData.my_role
            my_unum = strategyData.player_unum
            closest_unum = strategyData.get_closest_player_to_ball()

            # === If I'm the closest player to the ball → attack ===
            if my_unum == closest_unum:
                drawer.annotation((0, 10.5), f"{my_role}: Attacking!", drawer.Color.orange, f"{my_role}_attack")
                target_goal = (15, 0)
                return self.kickTarget(strategyData, strategyData.mypos, target_goal)

            # === Otherwise: off-ball forward → shadow striker run ===
            goal = np.array([15.0, 0.0])
            side = 1 if my_role == "LeftForward" else -1

            # Compute target slightly forward and inward toward goal
            shadow_target = strategyData.get_shadow_target(
                strategyData.ball_2d, goal,
                side=side,
                forward_offset=4.0,
                side_offset=1.0
            )

            # If very close to goal, tighten spacing
            if strategyData.ball_2d[0] > 10:
                shadow_target = strategyData.get_shadow_target(
                    strategyData.ball_2d, goal, 
                    side, 3.0, 0.5
                )

            drawer.annotation(
                (shadow_target[0], shadow_target[1] + 1.0),
                f"{my_role}: Shadow run",
                drawer.Color.orange,
                f"{my_role}_shadow"
            )

            return self.move(shadow_target, orientation=strategyData.ball_dir)

        # --- Final fallback: any closest player (forward, defender, or GK) can act ---
        if my_unum == closest_unum:
            drawer.annotation((0, 10.5), f"{strategyData.my_role}: Active ball owner", drawer.Color.cyan, "ball_owner")

            # 1️⃣ Check if shot is blocked
            if strategyData.is_shot_blocked(tolerance_angle=15, max_block_dist=6.0):
                pass_choice = strategyData.best_simple_pass_target()
                if pass_choice is not None:
                    _, pass_pos = pass_choice
                    return self.kickTarget(strategyData, strategyData.mypos, pass_pos)
                return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)

            # 2️⃣ If close to goal → shoot, else dribble
            dist_to_goal = np.linalg.norm(np.array([15, 0]) - np.array(strategyData.mypos))
            if dist_to_goal < 5:
                target_goal = (15, 0)
                drawer.annotation((0, 9.5), "Shooting!", drawer.Color.red, "shot_call")
                return self.kickTarget(strategyData, strategyData.mypos, target_goal)
            else:
                ball = np.array(strategyData.ball_2d)
                mypos = np.array(strategyData.mypos)
                goal_dir = np.array([15, 0]) - mypos
                goal_dir /= np.linalg.norm(goal_dir)
                dribble_target = ball + 0.5 * goal_dir
                drawer.line(mypos, dribble_target, 2, drawer.Color.cyan, "dribble_target")
                drawer.annotation((0, 9.5), "Walking Dribble", drawer.Color.cyan, "dribble_call")
                return self.move(dribble_target, orientation=strategyData.ball_dir)

        # Otherwise hold formation
        drawer.annotation((0, 10.5), "Holding Formation", drawer.Color.yellow, "status")
        return self.move(strategyData.my_desired_position, orientation=strategyData.my_desired_orientation)


    # ------------------------------------------------------------
    # 🟢 MVP DRIBBLING BEHAVIOUR (SAFE WALK-CARRY VERSION, RELAXED)
    # ------------------------------------------------------------
    def attempt_dribble(self, strategyData):
        """
        Light, stable dribble: walk gently through the ball instead of blasting it.
        Used when the player is the active ball owner in PlayOn.
        """
        drawer = self.world.draw
        mypos = strategyData.mypos
        ball = strategyData.ball_2d

        # Aim a short distance toward opponent goal (gentle nudge direction)
        direction = np.array([15, 0]) - np.array(mypos)
        direction = direction / np.linalg.norm(direction)
        dribble_target = np.array(ball) + 0.8 * direction  # move 0.8 m ahead of ball

        drawer.line(mypos, dribble_target, 2, drawer.Color.cyan, "dribble_target")
        drawer.annotation((0, 10.5), "Dribbling (walk-nudge)", drawer.Color.cyan, "status")

        # 👇 The core: walk gently through the ball instead of kicking
        return self.move(dribble_target, orientation=strategyData.ball_dir)


    def attempt_soft_kick_dribble(self, strategyData):
        """
        Stable MVP dribble: walk forward while keeping the ball close.
        No kick animation (prevents falling).
        """
        drawer = self.world.draw
        mypos = np.array(strategyData.mypos)
        ball = np.array(strategyData.ball_2d)

        # Direction toward opponent goal
        goal_dir = np.array([15, 0]) - mypos
        goal_dir = goal_dir / np.linalg.norm(goal_dir)

        # Keep ball slightly ahead of player
        dribble_target = ball + 0.5 * goal_dir

        drawer.line(mypos, dribble_target, 2, drawer.Color.cyan, "safe_dribble_target")
        drawer.annotation((0, 10.5), "Walking Dribble (Safe)", drawer.Color.cyan, "status")

        # Gentle walk that just keeps moving toward the ball
        return self.move(dribble_target, orientation=strategyData.ball_dir)


    #--------------------------------------- Fat proxy auxiliary methods

    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")