import pygame
import random
import math
import sys

#Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
BG_COLOR = (0, 0, 0)
WHITE = (255, 255, 255)
ZONE_BORDER_COLOR = (200, 200, 200)

# Colors for spawning logic
COLOR_RED = (255, 0, 0)
COLOR_YELLOW = (255, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)

# Robot settings
ROBOT_SPEED = 5.0
ROBOT_SIZE = 30
ROBOT_COLOR = (100, 100, 255) 

# Circle settings
CIRCLE_RADIUS_MIN = 10
CIRCLE_RADIUS_MAX = 20

# Zone Definitions
ZONE_WIDTH = 130
ZONE_HEIGHT = 130

#Classes

class Vector2:
    """Simple 2D vector class for movement logic calculations."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_tuple(self):
        return (self.x, self.y)
    
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        l = self.length()
        if l != 0:
            return Vector2(self.x / l, self.y / l)
        return Vector2(0, 0)

    def scale(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

class ColorClassifier:
    """k-Nearest Neighbors (KNN) logic for color classification."""
    def __init__(self, k=1):
        self.k = k
        self.training_data = [] # List of tuples: ((r, g, b), label)

    def train(self, color_samples):
        """
        Populate the knowledge base.
        color_samples: List of dictionaries or tuples {'color': (r,g,b), 'label': 'RED'}
        """
        for sample in color_samples:
            self.training_data.append((sample['color'], sample['label']))

    def predict(self, color):
        """
        Predicts the label for a given RGB color.
        Returns 'UNKNOWN' if confidence is low or distance is too high (optional),
        but basic KNN just returns closest.
        """
        if not self.training_data:
            return "UNKNOWN"

        distances = []
        r1, g1, b1 = color
        
        for train_color, label in self.training_data:
            r2, g2, b2 = train_color
            # Euclidean distance in RGB space
            dist = math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
            distances.append((dist, label))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[0])
        
        # Take closest k
        k_nearest = distances[:self.k]
        
        # If the nearest neighbor is too far away, classify as UNKNOWN
        # This handles the case of gray/random colors not matching our purified primitives
        if k_nearest[0][0] > 100: # Threshold for "closeness"
            return "UNKNOWN"

        # Majority vote (for k=1 it's just the first one)
        return k_nearest[0][1]

class CircleObject:
    """Represents a sortable circle object."""
    def __init__(self, start_pos=None, start_vel=None):
        self.radius = random.randint(CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX)
        
        if start_pos:
            self.pos = start_pos
        else:
            # Default random placement (safe margin)
            self.pos = Vector2(
                random.randint(150, SCREEN_WIDTH - 150),
                random.randint(150, SCREEN_HEIGHT - 150)
            )

        if start_vel:
             self.vel = start_vel
        else:
             self.vel = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)) # Add random slow movement

        self.color = self._generate_random_color()
        self.rect = pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)
        self.is_carried = False

    def _generate_random_color(self):
        """Generates Red, Yellow, Blue, Green in variations, or random noise."""
        roll = random.random()
        
        # Helper to add noise to a color
        def noisy(c):
            noise_level = 30
            r = min(255, max(0, c[0] + random.randint(-noise_level, noise_level)))
            g = min(255, max(0, c[1] + random.randint(-noise_level, noise_level)))
            b = min(255, max(0, c[2] + random.randint(-noise_level, noise_level)))
            return (r, g, b)

        if roll < 0.2:
            return noisy(COLOR_RED)
        elif roll < 0.4:
            return noisy(COLOR_YELLOW)
        elif roll < 0.6:
            return noisy(COLOR_BLUE)
        elif roll < 0.8:
            return noisy(COLOR_GREEN)
        else:
            # Random color
            return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def update_rect(self):
        # Physics update
        if not self.is_carried:
            self.pos = self.pos + self.vel
            
            # Boundary checks - bounce off walls
            if self.pos.x - self.radius < 0:
                self.pos.x = self.radius
                self.vel.x *= -1
            elif self.pos.x + self.radius > SCREEN_WIDTH:
                self.pos.x = SCREEN_WIDTH - self.radius
                self.vel.x *= -1
                
            if self.pos.y - self.radius < 0:
                self.pos.y = self.radius
                self.vel.y *= -1
            elif self.pos.y + self.radius > SCREEN_HEIGHT:
                self.pos.y = SCREEN_HEIGHT - self.radius
                self.vel.y *= -1

            # Friction (optional, keeps them from going forever if we wanted, but constant slow motion is fine for "living" feel)
            # self.vel = self.vel.scale(0.99)

        self.rect.center = (int(self.pos.x), int(self.pos.y))

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)
        pygame.draw.circle(surface, WHITE, (int(self.pos.x), int(self.pos.y)), self.radius, 1)

class DestinationZone:
    """Represents a drop-off zone."""
    def __init__(self, rect, color, label_text, zone_type):
        self.rect = rect
        self.color = color
        self.label_text = label_text
        self.zone_type = zone_type 
        self.font = pygame.font.SysFont("Arial", 16, bold=True)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, 2)
        
        text_surf = self.font.render(self.label_text, True, self.color)
        text_rect = text_surf.get_rect(center=(self.rect.centerx, self.rect.centery))
        surface.blit(text_surf, text_rect)

class Robot:
    """The autonomous robot."""
    def __init__(self, start_x, start_y, classifier):
        self.pos = Vector2(start_x, start_y)
        self.speed = ROBOT_SPEED
        self.carried_object = None
        self.state = "SCANNING" 
        self.target_circle = None
        self.target_zone = None
        self.classifier = classifier # Reference to the shared classifier
        self.rect = pygame.Rect(0, 0, ROBOT_SIZE, ROBOT_SIZE)
        self.rect.center = (int(self.pos.x), int(self.pos.y))
        self.waypoints = [] # List of Vector2 points

        # Stuck detection
        self.last_pos = Vector2(start_x, start_y)
        self.last_move_time = pygame.time.get_ticks()
        self.stuck_threshold = 100 # 0.1 seconds


    def get_avoidance_waypoint(self, target_pos, circles, zones):
        # 1. Circle Avoidance
        # Simple Raycast / Boxcast check
        to_target = target_pos - self.pos
        dist_to_target = to_target.length()
        if dist_to_target == 0:
            return None
        
        dir_to_target = to_target.normalize()
        
        # Check against all circles
        closest_obs = None
        min_dist = float('inf')
        
        # Width of the path (robot size + margin)
        path_width = ROBOT_SIZE + 20 
        
        for circle in circles:
            if circle.is_carried: continue # Don't avoid carried object
            
            # Vector from robot to circle center
            to_circle = circle.pos - self.pos
            
            # Project circle center onto the line of movement
            projection = (to_circle.x * dir_to_target.x) + (to_circle.y * dir_to_target.y)
            
            # If projection is negative (behind) or further than target, ignore
            if projection < 0 or projection > dist_to_target:
                continue
                
            # Distance from line
            closest_point_on_line = self.pos + dir_to_target.scale(projection)
            dist_from_line = (circle.pos - closest_point_on_line).length()
            
            if dist_from_line < (circle.radius + path_width/2):
                if projection < min_dist:
                    min_dist = projection
                    closest_obs = circle

        # Check against Zones (Territory Avoidance)
        # Avoid ALL zones that are not the target
        closest_zone_obs = None
        # Reuse min_dist logic or separate? Let's check zones intersecting the line.
        # Ray-AABB Intersection or simplified sampling
        
        for zone in zones:
            if zone == self.target_zone: continue # Don't avoid target
            
            # Inflate rect for safety margin
            margin = ROBOT_SIZE // 2 + 10
            safe_rect = zone.rect.inflate(margin*2, margin*2)
            
            # Check if line segment intersects rect
            if safe_rect.clipline(self.pos.to_tuple(), target_pos.to_tuple()):
                # It intersects. Find closest point or corner to navigate around.
                # Actually, clipline returns the segment inside.
                # For avoidance, we want to go via a corner.
                
                # Distance to center of zone?
                zone_center = Vector2(safe_rect.centerx, safe_rect.centery)
                dist = (zone_center - self.pos).length()
                
                if dist < min_dist: # Only avoid if it's the closest thing
                     min_dist = dist
                     closest_zone_obs = zone

        # Resolve Zone Collision first (usually bigger/static) or Circle?
        # Let's prioritize Zones as they are "Walls"
        
        if closest_zone_obs:
            # Navigate around the zone. 
            # Pick the corner that minimizes path length: Dist(Self->Corner) + Dist(Corner->Target)
            corners = [
                Vector2(closest_zone_obs.rect.left - 25, closest_zone_obs.rect.top - 25),
                Vector2(closest_zone_obs.rect.right + 25, closest_zone_obs.rect.top - 25),
                Vector2(closest_zone_obs.rect.left - 25, closest_zone_obs.rect.bottom + 25),
                Vector2(closest_zone_obs.rect.right + 25, closest_zone_obs.rect.bottom + 25)
            ]
            
            best_corner = None
            min_len = float('inf')
            
            for c in corners:
                # Check if this corner is reachable (simple check) 
                # or just pick the geometric best
                path_len = (c - self.pos).length() + (target_pos - c).length()
                if path_len < min_len:
                    min_len = path_len
                    best_corner = c
            
            # CLAMP CORNER to screen
            if best_corner:
                 best_corner.x = max(ROBOT_SIZE, min(SCREEN_WIDTH - ROBOT_SIZE, best_corner.x))
                 best_corner.y = max(ROBOT_SIZE, min(SCREEN_HEIGHT - ROBOT_SIZE, best_corner.y))

            return best_corner


        if closest_obs:
            # Found an obstacle. Calculate avoidance point.
            # We want to go to the side. Which side? The one it's already closer to, or purely based on tangent.
            # Vector from obstacle center to line
            to_circle = closest_obs.pos - self.pos
            projection = (to_circle.x * dir_to_target.x) + (to_circle.y * dir_to_target.y)
            closest_point_on_line = self.pos + dir_to_target.scale(projection)
            
            # Vector from line to obstacle center
            perp_vec = closest_obs.pos - closest_point_on_line
            
            # If perfectly on line, choose arbitrary right
            if perp_vec.length() < 0.1:
                perp_dir = Vector2(-dir_to_target.y, dir_to_target.x) # Rotate 90 deg
            else:
                perp_dir = perp_vec.normalize()
            
            avoid_dir = perp_dir.scale(-1)
            avoid_dist = closest_obs.radius + ROBOT_SIZE + 10
            
            # The point should be "around" the sphere. 
            # Let's project a point alongside the sphere at the point of closest approach
            avoidance_point = closest_point_on_line + avoid_dir.scale(avoid_dist + dist_from_line) 
            
            # CLAMP AVOIDANCE POINT
            avoidance_point.x = max(ROBOT_SIZE, min(SCREEN_WIDTH - ROBOT_SIZE, avoidance_point.x))
            avoidance_point.y = max(ROBOT_SIZE, min(SCREEN_HEIGHT - ROBOT_SIZE, avoidance_point.y))

            return avoidance_point

            
        return None

    def update(self, circles, zones):
        if self.state == "SCANNING":
            self.find_nearest_circle(circles)
            
        elif self.state == "MOVING_TO_CIRCLE":
            if self.target_circle not in circles: 
                self.state = "SCANNING"
                self.target_circle = None
                return

            dist = (self.target_circle.pos - self.pos).length()
            if dist < 5.0: 
                self.pick_up(self.target_circle)
                self.decide_destination(zones)
                self.state = "MOVING_TO_ZONE"
            else:
                self.move_towards(self.target_circle.pos)

        elif self.state == "MOVING_TO_ZONE":
            if not self.target_zone:
                self.state = "SCANNING"
                return

            # Waypoint navigation
            if not self.waypoints:
                # Should not happen in this logic, but recover
                 target_pos = Vector2(self.target_zone.rect.centerx, self.target_zone.rect.centery)
                 self.waypoints.append(target_pos)
            
            # Smart Recalculation Loop
            # 1. Check if we reached the current waypoint
            current_target = self.waypoints[0]
            dist = (current_target - self.pos).length()
            
            if dist < 5.0:
                self.waypoints.pop(0)
                if not self.waypoints:
                    # Reached final destination (Zone)
                    self.drop_object()
                    self.state = "SCANNING"
                return

            # 2. Check path to CURRENT waypoint
            avoid_pt = self.get_avoidance_waypoint(current_target, circles, zones)
            
            if avoid_pt:
                # Path to Current Waypoint (Center or Intermediate) is BLOCKED.
                # Optimization: Should we try to go straight to Final Destination instead?
                if len(self.waypoints) > 1:
                    final_dest = self.waypoints[-1]
                    # Check if direct line to Final is clear
                    avoid_pt_final = self.get_avoidance_waypoint(final_dest, circles, zones)
                    if not avoid_pt_final:
                        # Direct path is clear! Abandon Center, go Direct.
                        self.waypoints = [final_dest]
                        self.move_towards(final_dest)
                        return
                    
                # Direct optimization failed or not applicable.
                # Must avoid the obstacle on the current path.
                
                # Check if we assume "avoid_pt" is a new waypoint to INSERT
                # To prevent infinite insertion loop if the point is same, compare?
                # Usually fine as we pop close ones.
                
                self.waypoints.insert(0, avoid_pt)
                self.move_towards(avoid_pt)
            else:
                # Path to Current Waypoint is CLEAR.
                # Just follow it. Do NOT try to shortcut if not blocked (User wants "Middle" bias).
                self.move_towards(current_target)

        # STUCK DETECTION
        current_time = pygame.time.get_ticks()
        moved_dist = (self.pos - self.last_pos).length()
        
        if moved_dist > 2.0: # Arbitrary small movement threshold
             self.last_pos = Vector2(self.pos.x, self.pos.y)
             self.last_move_time = current_time
        elif self.state in ["MOVING_TO_CIRCLE", "MOVING_TO_ZONE"]:
             # Has not moved significantly
             if current_time - self.last_move_time > self.stuck_threshold:
                 print("Robot Stuck! Recalculating route...")
                 self.waypoints = [] # Clear waypoints to force logic reset
                 # If moving to zone, recalculation happens in next loop automatically?
                 # Actually, update logic for "MOVING_TO_ZONE" checks if not waypoints -> append target.
                 # But it doesn't clear "Current Target" if it thinks it's valid.
                 # By clearing waypoints, next frame: if MOVING_TO_ZONE, it re-adds target.
                 # If MOVING_TO_CIRCLE, it just continues move_towards.
                 
                 # To truly "shake" it, maybe move it slightly or just let the reset handle it.
                 # For "MOVING_TO_CIRCLE", we might want to re-scan or pick a new angle.
                 if self.state == "MOVING_TO_CIRCLE":
                      # Maybe just abort and re-scan
                      self.state = "SCANNING"
                      self.target_circle = None
                 else:
                      # Force re-pathing
                      self.last_move_time = current_time # Reset timer

        
        # Screen Clamping
        self.pos.x = max(ROBOT_SIZE//2, min(SCREEN_WIDTH - ROBOT_SIZE//2, self.pos.x))
        self.pos.y = max(ROBOT_SIZE//2, min(SCREEN_HEIGHT - ROBOT_SIZE//2, self.pos.y))

        self.rect.center = (int(self.pos.x), int(self.pos.y))
        
        if self.carried_object:
            self.carried_object.pos = Vector2(self.pos.x, self.pos.y - 10)
            self.carried_object.update_rect()

    def find_nearest_circle(self, circles):
        available_circles = [c for c in circles if not c.is_carried]
        if not available_circles:
            return 
        
        nearest = None
        min_dist = float('inf')
        
        for c in available_circles:
            dist = (c.pos - self.pos).length()
            if dist < min_dist:
                min_dist = dist
                nearest = c
        
        if nearest:
            self.target_circle = nearest
            self.state = "MOVING_TO_CIRCLE"

    def move_towards(self, target_pos):
        direction = target_pos - self.pos
        if direction.length() > 0:
            velocity = direction.normalize().scale(self.speed)
            self.pos = self.pos + velocity

    def pick_up(self, circle):
        self.carried_object = circle
        circle.is_carried = True

    def drop_object(self):
        self.carried_object.is_carried = False 
        self.carried_object = None

    def decide_destination(self, zones):
        # Use ML Classifier
        obj_color = self.carried_object.color
        predicted_label = self.classifier.predict(obj_color)
        
        print(f"Detected: {obj_color} -> Classified as: {predicted_label}") # Deubg output (visible in console)

        # Find matching zone
        found = False
        for zone in zones:
            if zone.zone_type == predicted_label:
                self.target_zone = zone
                found = True
                break
        
        if not found:
            for zone in zones:
                if zone.zone_type == "UNKNOWN":
                    self.target_zone = zone
                    break

        # Route via Center (Hybrid)
        if self.target_zone:
            # 1. Center Waypoint
            center_wp = Vector2(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            
            # 2. Zone Waypoint
            zone_wp = Vector2(self.target_zone.rect.centerx, self.target_zone.rect.centery)
            
            self.waypoints = [center_wp, zone_wp]

    def draw(self, surface):
        color = ROBOT_COLOR
        pygame.draw.rect(surface, color, (self.rect.left, self.rect.top, 8, self.rect.height))
        pygame.draw.rect(surface, color, (self.rect.right - 8, self.rect.top, 8, self.rect.height))
        pygame.draw.rect(surface, color, (self.rect.left, self.rect.bottom - 8, self.rect.width, 8))
        
        if self.state == "SCANNING":
            pygame.draw.circle(surface, (0, 255, 0), self.rect.center, 5)
            
        # Draw Route (Debug/Visual)
        if self.waypoints and len(self.waypoints) > 0:
            points = [(self.pos.x, self.pos.y)] + [wp.to_tuple() for wp in self.waypoints]
            if len(points) > 1:
                pygame.draw.lines(surface, (255, 255, 255), False, points, 2)

class GameManager:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("2D Color Sorting Machine (KNN ML)")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        self.font = pygame.font.SysFont("Arial", 16)
        
        # Initialize and Train Classifier
        self.classifier = ColorClassifier(k=1)
        self._train_classifier()

        self.circles = []
        self.zones = []
        self.robot = Robot(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, self.classifier)
        
        self.spawn_event = pygame.USEREVENT + 1
        pygame.time.set_timer(self.spawn_event, 5000)

        self._init_zones()
        self._spawn_initial_circles()

    def _train_classifier(self):
        # Feed the model prototypical examples
        training_data = [
            {'color': (255, 0, 0), 'label': 'RED'},
            {'color': (200, 50, 50), 'label': 'RED'}, # Darker Red
            
            {'color': (255, 255, 0), 'label': 'YELLOW'},
            {'color': (200, 200, 50), 'label': 'YELLOW'},
            
            {'color': (0, 0, 255), 'label': 'BLUE'},
            {'color': (50, 50, 200), 'label': 'BLUE'},
            
            {'color': (0, 255, 0), 'label': 'GREEN'},
            {'color': (50, 200, 50), 'label': 'GREEN'}
        ]
        self.classifier.train(training_data)
        print("Model Trained with basic color data.")

    def _init_zones(self):
        # 4 Corners + Middle Bottom for Unknown
        
        # Top-Left: YELLOW
        self.zones.append(DestinationZone(
            pygame.Rect(10, 10, ZONE_WIDTH, ZONE_HEIGHT), 
            COLOR_YELLOW, "YELLOW", "YELLOW"
        ))
        
        # Top-Right: RED
        self.zones.append(DestinationZone(
            pygame.Rect(SCREEN_WIDTH - ZONE_WIDTH - 10, 10, ZONE_WIDTH, ZONE_HEIGHT), 
            COLOR_RED, "RED", "RED"
        ))

        # Bottom-Left: BLUE
        self.zones.append(DestinationZone(
            pygame.Rect(10, SCREEN_HEIGHT - ZONE_HEIGHT - 10, ZONE_WIDTH, ZONE_HEIGHT), 
            COLOR_BLUE, "BLUE", "BLUE"
        ))

        # Bottom-Right: GREEN
        self.zones.append(DestinationZone(
            pygame.Rect(SCREEN_WIDTH - ZONE_WIDTH - 10, SCREEN_HEIGHT - ZONE_HEIGHT - 10, ZONE_WIDTH, ZONE_HEIGHT), 
            COLOR_GREEN, "GREEN", "GREEN"
        ))

        # Bottom-Middle: UNKNOWN
        self.zones.append(DestinationZone(
            pygame.Rect(SCREEN_WIDTH // 2 - ZONE_WIDTH // 2, SCREEN_HEIGHT - ZONE_HEIGHT - 10, ZONE_WIDTH, ZONE_HEIGHT), 
            WHITE, "UNKNOWN", "UNKNOWN"
        ))

    def _spawn_initial_circles(self):
        attempts = 0
        while len(self.circles) < 15 and attempts < 100:
            new_circle = CircleObject()
            if self._is_valid_spawn(new_circle):
                self.circles.append(new_circle)
            attempts += 1

    def _is_valid_spawn(self, circle):
        # Check collision with existing circles
        for c in self.circles:
            dist = (circle.pos - c.pos).length()
            if dist < (circle.radius + c.radius) + 5: # 5px buffer
                return False
        # Check collision with zones (don't spawn INSIDE a zone)
        for z in self.zones:
            if z.rect.colliderect(circle.rect):
                return False
        return True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == self.spawn_event:
                # Try to spawn from Left Side Direction
                for _ in range(5): 
                    # Left side gap: Between Top-Left Zone and Bottom-Left Zone
                    # Zone Height = 130. Screen Height = 600.
                    # Gap Y range approx 150 to 450.
                    spawn_y = random.randint(150, 450)
                    spawn_x = -30 # Start slightly off screen or just at edge
                    
                    start_pos = Vector2(spawn_x, spawn_y)
                    # Velocity: Move Right mostly, slight random Y
                    start_vel = Vector2(random.uniform(1.5, 3.0), random.uniform(-0.5, 0.5))
                    
                    new_circle = CircleObject(start_pos, start_vel)
                    
                    if self._is_valid_spawn(new_circle):
                        self.circles.append(new_circle)
                        break

    def update(self):
        self._resolve_collisions()
        self.robot.update(self.circles, self.zones)
        
        # Update circles physics
        for circle in self.circles:
            circle.update_rect()
        
    def _resolve_collisions(self):
        # Simple elastic collision between circles
        for i in range(len(self.circles)):
            for j in range(i + 1, len(self.circles)):
                c1 = self.circles[i]
                c2 = self.circles[j]
                
                # Don't collide if carried
                if c1.is_carried or c2.is_carried:
                    continue

                dist_vec = c1.pos - c2.pos
                dist = dist_vec.length()
                min_dist = c1.radius + c2.radius
                
                if dist < min_dist:
                    # 1. Resolve overlap (Static resolution)
                    overlap = min_dist - dist
                    if dist == 0: # Avoid division by zero if exact same pos
                        dist_vec = Vector2(1, 0)
                        dist = 1
                    
                    push_vec = dist_vec.normalize().scale(overlap / 2)
                    c1.pos = c1.pos + push_vec
                    c2.pos = c2.pos - push_vec
                    
                    # 2. Bounce (Dynamic resolution)
                    # Swap velocities (approximate for equal mass)
                    # For better physics: 1D elastic collision along the normal
                    normal = dist_vec.normalize()
                    
                    # Relative velocity
                    rel_vel = c1.vel - c2.vel
                    vel_along_normal = (rel_vel.x * normal.x + rel_vel.y * normal.y)
                    
                    # Do not resolve if velocities are separating
                    if vel_along_normal > 0:
                        continue
                        
                    # Restitution (bounciness)
                    e = 1.0 
                    j_impulse = -(1 + e) * vel_along_normal
                    j_impulse /= 2 # 1/mass1 + 1/mass2 (mass=1 for both)
                    
                    impulse = normal.scale(j_impulse)
                    c1.vel = c1.vel + impulse
                    c2.vel = c2.vel - impulse

        # Check delivery AND Zone Collision (Walls)
        balls_to_remove = []
        for circle in self.circles:
            if not circle.is_carried:
                for zone in self.zones:
                    # Check for collision with zone
                    if zone.rect.colliderect(circle.rect):
                        # Predict color to see if it belongs
                        predicted_label = self.classifier.predict(circle.color)
                        
                        if predicted_label == zone.zone_type:
                            # Correct zone (or at least matching classifier) -> Absorb
                            # Ensure it's mostly inside before absorbing to avoid edge clipping? 
                            # Actually, standard behavior is usually collidepoint center
                            if zone.rect.collidepoint(circle.pos.to_tuple()):
                                balls_to_remove.append(circle)
                                break
                        else:
                            # Wrong zone -> BOUNCE
                            # Treat zone rect as a static AABB for collision
                            
                            # Determine penetration depth and normal
                            # This is a bit tricky with Circle vs Rect, but AABB vs AABB approximation or Clamp is easier
                            
                            # Find closest point on rect to circle center
                            closest_x = max(zone.rect.left, min(circle.pos.x, zone.rect.right))
                            closest_y = max(zone.rect.top, min(circle.pos.y, zone.rect.bottom))
                            
                            distance_x = circle.pos.x - closest_x
                            distance_y = circle.pos.y - closest_y
                            
                            # If distance is zero, center is inside, but we checked that above.
                            # However, we only absorb if center is inside. If center is outside but radius touches...
                            
                            dist_sq = distance_x**2 + distance_y**2
                            
                            if dist_sq < circle.radius**2:
                                # Collision confirmed
                                # Push out
                                
                                # Simple box logic: see which face is closest
                                d_left = abs(circle.pos.x - zone.rect.left)
                                d_right = abs(circle.pos.x - zone.rect.right)
                                d_top = abs(circle.pos.y - zone.rect.top)
                                d_bottom = abs(circle.pos.y - zone.rect.bottom)
                                
                                min_d = min(d_left, d_right, d_top, d_bottom)
                                
                                if min_d == d_left:
                                    circle.pos.x = zone.rect.left - circle.radius
                                    circle.vel.x = -abs(circle.vel.x)
                                elif min_d == d_right:
                                    circle.pos.x = zone.rect.right + circle.radius
                                    circle.vel.x = abs(circle.vel.x)
                                elif min_d == d_top:
                                    circle.pos.y = zone.rect.top - circle.radius
                                    circle.vel.y = -abs(circle.vel.y)
                                elif min_d == d_bottom:
                                    circle.pos.y = zone.rect.bottom + circle.radius
                                    circle.vel.y = abs(circle.vel.y)
        
        for b in balls_to_remove:
            if b in self.circles:
                self.circles.remove(b)

    def draw(self):
        self.screen.fill(BG_COLOR)
        
        for zone in self.zones:
            zone.draw(self.screen)
            
        for circle in self.circles:
            circle.draw(self.screen)
            
        self.robot.draw(self.screen)

        info_text = f"Circles: {len(self.circles)}"
        text_surf = self.font.render(info_text, True, WHITE)
        self.screen.blit(text_surf, (SCREEN_WIDTH // 2 - 40, 10))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GameManager()
    game.run()
