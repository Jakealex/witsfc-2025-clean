import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M


class Strategy():
    def __init__(self, world):
        self.world = world
        self.play_mode = world.play_mode
        self.robot_model = world.robot

        self.PM_GROUP = getattr(world, 'pm_group', None)
        if self.PM_GROUP is None:
            self.PM_GROUP = getattr(world, 'play_mode_group', None)
        if self.PM_GROUP is None:
            mg_active = getattr(world, 'MG_ACTIVE_BEAM', None)
            mg_passive = getattr(world, 'MG_PASSIVE_BEAM', None)
            try:
                if self.play_mode == getattr(world, 'M_BEFORE_KICKOFF', None) and mg_passive is not None:
                    self.PM_GROUP = mg_passive
                else:
                    self.PM_GROUP = mg_active if mg_active is not None else mg_passive
            except Exception:
                self.PM_GROUP = None
        
        # ===== SAFE INITIALIZATION WITH HELPER =====
        # ...existing code...
        # ===== SAFE INITIALIZATION WITH HELPER =====
        def safe_pos_2d(obj, attr='loc_head_position', default=(0.0, 0.0)):
            """Extract 2D position safely, return numpy array."""
            try:
                # Avoid using obj in boolean context since obj may contain numpy arrays
                if obj is None:
                    return np.array(default, dtype=np.float64)
                val = getattr(obj, attr, None)
                if val is None:
                    return np.array(default, dtype=np.float64)
                # Ensure we have at least two elements
                if len(val) >= 2:
                    return np.asarray(val[:2], dtype=np.float64)
            except (AttributeError, TypeError, IndexError):
                pass
            return np.array(default, dtype=np.float64)
        
        # Core positions
        self.my_head_pos_2d = safe_pos_2d(self.robot_model, 'loc_head_position')
        self.player_unum = self.robot_model.unum
        
        # My position from teammates array
        try:
            self.mypos = safe_pos_2d(
                world.teammates[self.player_unum - 1], 
                'state_abs_pos'
            )
        except (TypeError, IndexError):
            self.mypos = np.array([0.0, 0.0], dtype=np.float64)
        
        # Team side: +1 = left, -1 = right
        self.side = 1 if world.team_side_is_left else -1
        
        # ===== SAFE POSITION EXTRACTION =====
        def extract_positions(players):
            """Safely extract list of numpy 2D positions from player list."""
            if players is None:
                return [None] * 5
            positions = []
            for p in players:
                if p is None:
                    positions.append(None)
                    continue
                state = getattr(p, 'state_abs_pos', None)
                if state is None:
                    positions.append(None)
                else:
                    positions.append(safe_pos_2d(p, 'state_abs_pos'))
            return positions
        
        self.teammate_positions = extract_positions(world.teammates)
        self.opponent_positions = extract_positions(world.opponents)
        
        # Optional tracking (unused, but kept for compatibility)
        self.team_dist_to_ball = None
        self.team_dist_to_oppGoal = None
        self.opp_dist_to_ball = None
        self.prev_important_positions_and_values = None
        self.curr_important_positions_and_values = None
        self.point_preferences = None
        self.combined_threat_and_definedPositions = None
        
        # Orientation
        self.my_ori = float(self.robot_model.imu_torso_orientation or 0.0)
        
        # Ball state - direct access since ball_abs_pos is not an object attribute

        # ...existing code...
        # Ball state - direct access since ball_abs_pos is not an object attribute
        # Safely extract 2D ball position, a smoothed/alias position (slow_ball_pos),
        # and a ball-facing direction for use elsewhere in the strategy.
        ball_raw = getattr(world, 'ball_abs_pos', None)
        if ball_raw is None:
            self.ball_2d = np.array([0.0, 0.0], dtype=np.float64)
        else:
            try:
                self.ball_2d = np.asarray(ball_raw[:2], dtype=np.float64)
            except Exception:
                self.ball_2d = np.array([0.0, 0.0], dtype=np.float64)

        # slow_ball_pos is used for distance checks — use current ball if no smoother is available
        self.slow_ball_pos = self.ball_2d.copy()

        # ball_dir: angle (degrees) pointing from my head toward the ball (safe fallback = 0.0)
        try:
            ball_vec = self.ball_2d - self.my_head_pos_2d
            self.ball_dir = float(M.vector_angle(ball_vec))
        except Exception:
            self.ball_dir = 0.0
# ...existing code...
# ...existing code...
        # ===== DISTANCE CALCULATIONS =====
        def compute_ball_distances(players):
            """Compute squared distances from players to slow_ball_pos."""
            if not players:
                return [1000.0] * 5
            
            distances = []
            for p in players:
                if p is None:
                    distances.append(1000.0)
                    continue
                state_pos = getattr(p, 'state_abs_pos', None)
                last_update = getattr(p, 'state_last_update', 0)
                fallen = getattr(p, 'state_fallen', False)
                is_self = getattr(p, 'is_self', False)
                
                valid = (state_pos is not None and last_update != 0 and
                         (world.time_local_ms - last_update <= 360 or is_self) and
                         not fallen)
                
                if valid:
                    pos = safe_pos_2d(p, 'state_abs_pos')
                    distances.append(float(np.sum((pos - self.slow_ball_pos) ** 2)))
                else:
                    distances.append(1000.0)
            return distances
        
        self.teammates_ball_sq_dist = compute_ball_distances(world.teammates)
        self.opponents_ball_sq_dist = compute_ball_distances(world.opponents)
# ...existing code...
        
        # Minimum distances and active player
        if self.teammates_ball_sq_dist:
            self.min_teammate_ball_sq_dist = min(self.teammates_ball_sq_dist)
            self.min_teammate_ball_dist = math.sqrt(self.min_teammate_ball_sq_dist)
            try:
                self.active_player_unum = self.teammates_ball_sq_dist.index(
                    self.min_teammate_ball_sq_dist) + 1
            except ValueError:
                self.active_player_unum = self.player_unum
        else:
            self.min_teammate_ball_sq_dist = 1000.0
            self.min_teammate_ball_dist = math.sqrt(1000.0)
            self.active_player_unum = self.player_unum
        
        if self.opponents_ball_sq_dist:
            self.min_opponent_ball_dist = math.sqrt(min(self.opponents_ball_sq_dist))
        else:
            self.min_opponent_ball_dist = math.sqrt(1000.0)
        
        # Desired state
        self.my_desired_position = self.mypos.copy()
        self.my_desired_orientation = self.ball_dir

    def GenerateTeamToTargetDistanceArray(self, target, world):
        for teammate in world.teammates:
            pass

    def IsFormationReady(self, point_preferences):
        """Check if teammates are in formation positions."""
        threshold_sq = 0.09  # 0.3m squared (FIXED: was incorrectly 0.3 in previous version)
        for i in range(1, 6):
            if i == self.active_player_unum:
                continue
            
            pos = self.teammate_positions[i - 1]
            if pos is None or i not in point_preferences:
                continue
            
            target = point_preferences[i]
            if isinstance(target, np.ndarray):
                dist_sq = np.sum((pos - target) ** 2)
                if dist_sq > threshold_sq:
                    return False
        return True

    def GetDirectionRelativeToMyPositionAndTarget(self, target):
        """Calculate direction to target relative to my position."""
        target_arr = np.asarray(target, dtype=np.float64)
        target_vec = target_arr - self.my_head_pos_2d
        target_dir = M.vector_angle(target_vec)
        return target_dir
    
    # === Submission 2 Helper Functions ===
    
    def get_closest_player_to_ball(self):
        """
        Returns the uniform number (unum) of the teammate closest to the ball.
        Uses the precomputed teammate-ball distances.
        """
        return self.active_player_unum
    
    def handle_play_mode(self):
        """
        Determines what each player should do depending on the play mode.
        Returns a 2D target position for move().
        For now this is simple logic you can expand later.
        """
        # Default: stand still at current desired position
        target = self.my_desired_position
        
        # --- Before Kickoff ---
        if self.play_mode == self.world.M_BEFORE_KICKOFF:
            # stay in formation; maybe face the ball
            return self.my_desired_position
        
        # --- Kickoff (left/right) ---
        elif self.play_mode in [
            self.world.M_OUR_KICKOFF,
            self.world.M_THEIR_KICKOFF,
        ]:
            # Defensive team stays in place, attacking team gets ready to move
            return self.my_desired_position
        
        # --- Goal scored or Game Over ---
        elif self.play_mode in [
            self.world.M_OUR_GOAL,
            self.world.M_THEIR_GOAL,
            self.world.M_GAME_OVER,
        ]:
            # Return to base formation
            return self.my_desired_position
        
        # --- Default (PlayOn etc.) ---
        else:
            return self.my_desired_position
    
    def generate_dynamic_formation(self):
        """
        1–1–3 diamond-ish:
        [GK], [CB], [LF, CF, RF]
        Adjust depth by ball_x.
        """
        ball_x = self.ball_2d[0]
        
        # --- Defensive (ball near our goal) ---
        if ball_x < -5:
            formation = [
                np.array([-13,  0]),  # 0 GK
                np.array([-11,  0]),  # 1 CenterBack
                np.array([ -6, -3]),  # 2 LeftForward (deeper)
                np.array([ -5,  0]),  # 3 CenterForward (deeper)
                np.array([ -6,  3]),  # 4 RightForward (deeper)
            ]
        
        # --- Balanced (midfield) ---
        elif -5 <= ball_x <= 5:
            formation = [
                np.array([-12,  0]),  # 0 GK
                np.array([ -8,  0]),  # 1 CenterBack
                np.array([ -1, -3]),  # 2 LeftForward
                np.array([  0,  0]),  # 3 CenterForward
                np.array([ -1,  3]),  # 4 RightForward
            ]
        
        # --- Attacking (ball near opponent goal) ---
        else:
            formation = [
                np.array([-10,  0]),  # 0 GK
                np.array([ -4,  0]),  # 1 CenterBack (holds midline)
                np.array([  4, -3]),  # 2 LeftForward
                np.array([  7,  0]),  # 3 CenterForward (highest)
                np.array([  4,  3]),  # 4 RightForward
            ]
        
        return formation
    
    # Map formation indices → readable role names
    ROLE_NAMES = ["Goalkeeper", "LeftDefender", "RightDefender", "LeftForward", "RightForward"]
    
    def get_role_name_from_position(self, assigned_pos, formation_positions):
        """
        Given my assigned formation position and the formation list,
        find which slot I match and return the human-friendly role name.
        """
        assigned_arr = np.asarray(assigned_pos, dtype=np.float64)
        for idx, pos in enumerate(formation_positions):
            # positions are numpy arrays; use allclose to be safe
            if np.allclose(pos, assigned_arr):
                return self.ROLE_NAMES[idx]
        return "Unknown"
    
    def best_simple_pass_target(self):
        """
        Find best pass target with opponent pressure consideration.
        Returns (unum, position) or None.
        """
        if not self.teammate_positions:
            return None
        
        my = self.mypos
        mode = self.play_mode
        W = self.world
        candidates = []
        
        for i, pos in enumerate(self.teammate_positions, start=1):
            if pos is None or i == self.player_unum:
                continue
            
            dist = np.linalg.norm(pos - my)
            if not np.isfinite(dist) or dist < 0.5:
                continue
            
            dx, dy = pos[0] - my[0], abs(pos[1] - my[1])
            
            # Check opponent pressure (NEW FEATURE)
            pressure = self._count_opponents_near(pos, radius=3.0)
            
            # Set-play logic
            if mode in [W.M_OUR_FREE_KICK, W.M_OUR_GOAL_KICK, 
                        W.M_OUR_CORNER_KICK, W.M_OUR_KICKOFF]:
                if dist <= 8 and dx > -1:
                    score = 50 - dist - 0.5 * dy - 10 * pressure  # Penalize pressure
                    if np.isfinite(score):
                        candidates.append((score, i, pos))
            
            # Normal play logic
            else:
                if dx > 0 and dy < 8:
                    score = dx - 0.2 * dist - 5 * pressure  # Penalize pressure
                    if np.isfinite(score):
                        candidates.append((score, i, pos))
        
        if not candidates:
            return None
        
        candidates.sort(reverse=True)
        return candidates[0][1], candidates[0][2]
    
    def _count_opponents_near(self, position, radius=3.0):
        """Count opponents within radius of position (helper for pass logic)."""
        if not self.opponent_positions:
            return 0
        
        count = 0
        pos = np.asarray(position, dtype=np.float64)
        for opp in self.opponent_positions:
            if opp is not None:
                dist = np.linalg.norm(opp - pos)
                if np.isfinite(dist) and dist < radius:
                    count += 1
        return count
    
    def get_mode_action(self):
        """
        Returns a short string describing the general team action
        for the current play mode.
        """
        W = self.world
        mode = self.play_mode
        
        # Reset / setup
        if mode in [W.M_BEFORE_KICKOFF, W.M_OUR_GOAL, W.M_THEIR_GOAL]:
            return "beam"
        
        # Our attacking restarts
        if mode in [
            W.M_OUR_KICKOFF, W.M_OUR_KICK_IN, W.M_OUR_CORNER_KICK,
            W.M_OUR_GOAL_KICK, W.M_OUR_FREE_KICK, W.M_OUR_PASS,
            W.M_OUR_DIR_FREE_KICK,
        ]:
            return "setplay"
        
        # Their restarts
        if mode in [
            W.M_THEIR_KICKOFF, W.M_THEIR_KICK_IN, W.M_THEIR_CORNER_KICK,
            W.M_THEIR_GOAL_KICK, W.M_THEIR_FREE_KICK, W.M_THEIR_PASS,
            W.M_THEIR_DIR_FREE_KICK,
        ]:
            return "defend"
        
        if mode == W.M_PLAY_ON:
            return "play"
        
        if mode == W.M_GAME_OVER:
            return "idle"
        
        return "idle"
    
    def get_setplay_positions(self):
        """
        Returns a list of 5 (x, y) np.array positions depending on the current play_mode.
        Used to position players intelligently for corners / free kicks / etc.
        """
        W = self.world
        mode = self.play_mode
        
        # --- OUR CORNER KICK (attacking corner) ---
        if mode == W.M_OUR_CORNER_KICK:
            return [
                np.array([-13, 0]),   # GK stays home
                np.array([10, -2]),   # option 1
                np.array([10,  2]),   # option 2
                np.array([13,  3]),   # forward near box
                np.array([0,   0]),   # holding midfielder
            ]
        
        # --- THEIR CORNER KICK (defensive corner) ---
        if mode == W.M_THEIR_CORNER_KICK:
            return [
                np.array([-13, 0]),   # GK
                np.array([-12, -3]),  # mark left
                np.array([-12,  3]),  # mark right
                np.array([-8,  -2]),  # cover short
                np.array([-6,   0]),  # edge of box / outlet
            ]
        
        # --- OUR FREE KICK / GOAL KICK ---
        if mode in [W.M_OUR_FREE_KICK, W.M_OUR_GOAL_KICK]:
            return [
                np.array([-13, 0]),   # GK or kicker
                np.array([-6, -4]),   # left outlet
                np.array([-6,  4]),   # right outlet
                np.array([-2,  0]),   # central receiver
                np.array([2,   0]),   # striker waiting
            ]
        
        # --- THEIR FREE KICK / GOAL KICK ---
        if mode in [W.M_THEIR_FREE_KICK, W.M_THEIR_GOAL_KICK]:
            return [
                np.array([-13, 0]),   # GK
                np.array([-10, -3]),  # left defender
                np.array([-10,  3]),  # right defender
                np.array([-6,  -1]),  # midfielder
                np.array([-4,   1]),  # midfielder
            ]
        
        # --- OUR KICKOFF ---
        if mode == W.M_OUR_KICKOFF:
            return [
                np.array([-13, 0]),
                np.array([-3, -2]),
                np.array([-3,  2]),
                np.array([0,  -1]),
                np.array([0,   1]),
            ]
        
        # --- THEIR KICKOFF ---
        if mode == W.M_THEIR_KICKOFF:
            return [
                np.array([-13, 0]),
                np.array([-7, -3]),
                np.array([-7,  3]),
                np.array([-4,  -1]),
                np.array([-4,   1]),
            ]
        
        # Default fallback — use current dynamic formation
        return self.generate_dynamic_formation()
    
    def is_shot_blocked(self, tolerance_angle=15, max_block_dist=6.0):
        """Check if opponent blocks shot path to goal."""
        if not self.opponent_positions:
            return False
        
        goal = np.array([15.0, 0.0], dtype=np.float64)
        goal_vec = goal - self.mypos
        goal_dir = M.vector_angle(goal_vec)
        goal_dist = np.linalg.norm(goal_vec)
        
        if not np.isfinite(goal_dist):
            return False
        
        for opp in self.opponent_positions:
            if opp is None:
                continue
            
            opp_vec = opp - self.mypos
            opp_dist = np.linalg.norm(opp_vec)
            
            if not np.isfinite(opp_dist):
                continue
            
            if opp_dist > goal_dist or opp_dist > max_block_dist:
                continue
            
            opp_dir = M.vector_angle(opp_vec)
            ang_diff = abs(M.normalize_deg(goal_dir - opp_dir))
            
            if ang_diff < tolerance_angle:
                return True
        
        return False
    
    def get_shadow_target(self, ball, goal, side=1, forward_offset=4.0, side_offset=1.0):
        """
        Computes a simple target position for the off-ball forward
        to crash toward goal while staying slightly offset sideways.
        
        Parameters
        ----------
        ball : (x, y) iterable
            Current ball position.
        goal : (x, y) iterable
            Goal target (usually (15, 0)).
        side : int
            +1 for left forward, -1 for right forward.
        forward_offset : float
            How far ahead of the ball (toward goal) the player should aim.
        side_offset : float
            How far to the side of the ball path to stay.
        
        Returns
        -------
        numpy.ndarray
            2D target coordinate for move().
        """
        ball = np.asarray(ball, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        
        dir_to_goal = goal - ball
        norm = np.linalg.norm(dir_to_goal)
        
        if norm < 1e-6:
            return ball  # avoid division by zero if goal == ball
        
        dir_to_goal /= norm
        dir_perp = np.array([-dir_to_goal[1], dir_to_goal[0]], dtype=np.float64)  # 90° rotated
        
        target = ball + forward_offset * dir_to_goal + side_offset * side * dir_perp
        
        return target
