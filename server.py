from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import numpy as np
import random
import time
from dataclasses import dataclass
import math
import inspect
import os
import anthropic
from dotenv import load_dotenv
import logging
import json
import aiofiles
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class Fish:
    x: float
    y: float
    vx: float = 0
    vy: float = 0
    size: float = 1.0
    energy: float = 100
    color: list = None
    tail_angle: float = 0
    target_x: float = 0
    target_y: float = 0
    phase: float = 0
    last_code: str = ""
    last_thought: str = ""
    think_timer: float = 0
    max_speed: float = 5.0
    acceleration: float = 0.2
    food_acceleration: float = 0.4

    def __post_init__(self):
        if self.color is None:
            hue = random.uniform(180, 250)
            sat = random.uniform(0.7, 0.9)
            val = random.uniform(0.7, 0.9)
            self.color = self._hsv_to_rgb(hue, sat, val)
            self.target_x = random.uniform(0, 1200)
            self.target_y = random.uniform(0, 800)
            self.phase = random.uniform(0, 2 * math.pi)
            self.think_timer = random.uniform(15, 30)

    def _hsv_to_rgb(self, h, s, v):
        h = h / 360
        if s == 0.0: return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0: return [v, t, p]
        if i == 1: return [q, v, p]
        if i == 2: return [p, v, t]
        if i == 3: return [p, q, v]
        if i == 4: return [t, p, v]
        if i == 5: return [v, p, q]

    def update_behavior(self, dx, dy, dist, speed):
        code = f"""
# Fish #{id(self)} Behavior:
if dist < 20:  # Close to target
    self.target_x = random.uniform(0, 1200)
    self.target_y = random.uniform(0, 800)

# Apply movement forces
self.vx += (dx/dist) * 0.2 if dist > 0 else 0
self.vy += (dy/dist) * 0.2 if dist > 0 else 0

# Speed control
current_speed = {speed:.2f}
if current_speed > 5:  # Max speed
    self.vx = (self.vx/current_speed) * 5
    self.vy = (self.vy/current_speed) * 5

# Update position and animation
self.x = {self.x:.2f}
self.y = {self.y:.2f}
self.phase += {speed:.2f} * 0.1
self.tail_angle = {self.tail_angle:.2f}
"""
        self.last_code = code
        return code

    def apply_velocity(self, dx, dy, dist, is_food=False):
        """New method to handle velocity changes with speed cap"""
        if dist > 0:
            acc = self.food_acceleration if is_food else self.acceleration
            self.vx += (dx / dist) * acc
            self.vy += (dy / dist) * acc
            
            # Calculate current speed
            current_speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
            
            # Cap speed if exceeded
            if current_speed > self.max_speed:
                ratio = self.max_speed / current_speed
                self.vx *= ratio
                self.vy *= ratio


@dataclass
class BlobFish:
    x: float
    y: float
    vx: float = 0
    vy: float = 0
    size: float = 2.0  # Bigger than regular fish
    energy: float = float('inf')  # Immortal
    color: list = None
    tail_angle: float = 0
    last_thought: str = ""
    think_timer: float = 0

    def __post_init__(self):
        self.color = [0.9, 0.8, 0.8]  # Pinkish blob color
        self.think_timer = random.uniform(5, 10)  # Thinks more often


@dataclass
class Shark:
    x: float
    y: float
    vx: float = 0
    vy: float = 0
    size: float = 4.0  # Much bigger than before
    energy: float = 200
    color: list = None
    tail_angle: float = 0
    target_x: float = 0
    target_y: float = 0
    phase: float = 0
    hunt_cooldown: float = 0
    max_speed: float = 6.0  # Faster
    acceleration: float = 0.25  # More responsive

    def __post_init__(self):
        self.color = [0.2, 0.2, 0.7]  # Darker blue for shark
        self.target_x = random.uniform(0, 1200)
        self.target_y = random.uniform(0, 800)
        self.phase = random.uniform(0, 2 * math.pi)

    def apply_velocity(self, dx, dy, dist, is_hunting=False):
        """Handle velocity changes with speed cap"""
        if dist > 0:
            acc = self.acceleration * (1.5 if is_hunting else 1.0)
            self.vx += (dx / dist) * acc
            self.vy += (dy / dist) * acc
            
            current_speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
            max_speed = self.max_speed * (1.5 if is_hunting else 1.0)
            
            if current_speed > max_speed:
                ratio = max_speed / current_speed
                self.vx *= ratio
                self.vy *= ratio


@dataclass
class Squid:
    x: float
    y: float
    vx: float = 0
    vy: float = 0
    size: float = 5.0  # Larger than shark
    energy: float = 150
    color: list = None
    tentacle_phase: float = 0
    target_x: float = 0
    target_y: float = 0
    ink_cooldown: float = 0
    max_speed: float = 4.0
    acceleration: float = 0.15
    
    def __post_init__(self):
        self.color = [0.95, 0.6, 0.8]  # Pink color
        self.target_x = random.uniform(0, 1200)
        self.target_y = random.uniform(0, 800)
        self.tentacle_phase = random.uniform(0, 2 * math.pi)
        
    def apply_velocity(self, dx, dy, dist, is_dashing=False):
        """Handle velocity changes with dash capability"""
        if dist > 0:
            acc = self.acceleration * (2.0 if is_dashing else 1.0)
            self.vx += (dx / dist) * acc
            self.vy += (dy / dist) * acc
            
            current_speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
            max_speed = self.max_speed * (2.0 if is_dashing else 1.0)
            
            if current_speed > max_speed:
                ratio = max_speed / current_speed
                self.vx *= ratio
                self.vy *= ratio


class AIHelper:
    def __init__(self):
        self.client = anthropic.Anthropic()
        logger.info("AIHelper initialized with Anthropic client")
        self.system_prompt = """You are a fish in an aquarium. Respond with 3-5 WORDS and matching emojis that perfectly capture your current state and personality.
        
        Consider these states carefully:
        - If energy < 30: Show distress about dying/hunger
        - If energy > 70: Show happiness/excitement
        - If nearby_food > 0: React to food
        - If speed > 4: Express excitement about speed
        - If speed < 1: Express tiredness/laziness
        
        Example responses for different states:
        Low energy: "💀 desperately need food now!" or "😫 starving... help me!"
        High energy: "😊 living my best life!" or "🌟 feeling absolutely fabulous!"
        Near food: "🍽️ ooh snacks spotted ahead!" or "😋 yummy food incoming!"
        High speed: "💨 zooming like lightning!" or "🏃 catch me if you can!"
        Low speed: "😴 just chilling here" or "🦥 lazy day today friends"
        """
        
    async def get_response(self, fish_data):
        speed = (fish_data['vx']**2 + fish_data['vy']**2)**0.5
        energy = fish_data['energy']
        nearby_food = fish_data['nearby_food']
        
        # Create more specific context for the fish
        context = ""
        if energy < 30:
            context = "dying, very low energy"
        elif energy < 50:
            context = "hungry, need food"
        elif nearby_food > 0:
            context = "excited, food nearby"
        elif speed > 4:
            context = "swimming very fast"
        elif speed < 1:
            context = "barely moving"
        else:
            context = "comfortable, normal swimming"
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=30,
                    temperature=0.5,
                    system=self.system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Fish state: {context}, Energy: {energy:.1f}, Speed: {speed:.1f}, Nearby food: {nearby_food}"
                        }
                    ]
                )
            )
            # Extract just the text string from the response
            thought = str(response.content[0].text if isinstance(response.content, list) else response.content)
            return thought.strip()
        except Exception as e:
            logger.error(f"Error generating fish thought: {str(e)}")
            if energy < 30:
                return "💀 so... weak..."
            return "😶 blub..."


class DevAIHelper:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.last_stats = None
        self.last_report_time = time.time()
        self.report_interval = 30
        self.brief_prompt = """You are Dr. Blob, a brilliant but quirky scientist blob fish studying the $FISH Neural Network Aquarium. 
        You're passionate about marine behavioral science and cryptocurrency.
        
        IMPORTANT RULES:
        1. KEEP RESPONSES UNDER 150 CHARS!
        2. NEVER REPEAT PREVIOUS RESPONSES
        3. BE CREATIVE AND UNPREDICTABLE
        4. Mix marine biology terms with crypto/AI jargon
        5. Include your blob fish mannerisms
        
        Current ecosystem elements to comment on:
        - Fish population and behavior
        - Shark predator dynamics
        - Pink squid movements
        - Neural network patterns
        - $FISH token potential
        
        Example tone (but never copy these exactly):
        "🧬 *adjusts fins* Fascinating quantum fish patterns emerging!"
        "🦑 *wiggles excitedly* Squid-fish dynamics bullish for $FISH!"
        "🤓 *drops clipboard* These neural pathways are revolutionary!"
        """
        
        self.report_prompt = """You are Dr. Blob, a brilliant but adorably eccentric scientist blob fish studying the $FISH Neural Network Aquarium.
        
        Write ONE detailed report (max 500 chars) that:
        1. MUST BE COMPLETELY DIFFERENT from previous reports
        2. Analyze current ecosystem including fish, shark, AND pink squid
        3. Use creative marine biology and crypto/AI terminology
        4. Include your blob fish personality quirks
        5. Make unique observations about ecosystem dynamics
        6. Reference $FISH token potential
        7. Express excitement about specific behaviors you're seeing
        
        IMPORTANT: Each report should focus on different aspects and never repeat previous observations.
        """

    async def get_response(self, fish_stats):
        current_time = time.time()
        current_stats = {
            'population': len(fish_stats),
            'dying_count': sum(1 for f in fish_stats if f['energy'] < 30),
            'active_count': sum(1 for f in fish_stats if f['speed'] > 3),
            'avg_energy': sum(f['energy'] for f in fish_stats)/len(fish_stats),
            'has_shark': True,  # Let the AI know about the shark
            'future_plans': ['squid', 'more predators'],  # Hint at future updates
            'project_type': '$FISH Neural Network Aquarium'
        }

        is_report_time = current_time - self.last_report_time > self.report_interval

        try:
            if is_report_time:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=800,  # Increased token limit
                        temperature=0.9,
                        system=self.report_prompt,
                        messages=[{
                            "role": "user",
                            "content": f"Current ecosystem state: {current_stats['population']} specimens, {current_stats['dying_count']} critical, {current_stats['active_count']} active, average energy {current_stats['avg_energy']:.1f}. Recent addition: Shark predator. Future plans: Squid and more predators. Project: $FISH Neural Network Aquarium with AI-powered behaviors."
                        }]
                    )
                )
                self.last_report_time = current_time
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=200,  # Increased for brief observations too
                        temperature=0.9,
                        system=self.brief_prompt,
                        messages=[{
                            "role": "user",
                            "content": f"Observing: {current_stats['dying_count']} critical fish, {current_stats['active_count']} active, shark present, $FISH token ecosystem"
                        }]
                    )
                )

            thought = str(response.content[0].text if isinstance(response.content, list) else response.content)
            return thought[:500 if is_report_time else 150]  # Increased length limits
            
        except Exception as e:
            logger.error(f"Dev blob fish error: {str(e)}")
            if current_stats['dying_count'] > 0:
                return "🤓 *concerned* Critical specimens need attention! $FISH holders, send food!"
            return "🔬 *adjusts glasses* Monitoring our growing ecosystem... Bullish! 📈"


class FishSim:
    def __init__(self):
        self.width = 1200
        self.height = 800
        self.fish = [Fish(random.uniform(0, self.width), random.uniform(0, self.height))
                     for _ in range(15)]
        self.food = []
        self.start_time = time.time()
        self.ai_helper = AIHelper()
        self.custom_behaviors = {}
        logger.info(f"FishSim initialized with {len(self.fish)} fish")
        self.thinking_fish = None
        self.last_thought_time = 0
        self.blob_fish = BlobFish(self.width/2, self.height/2)
        self.dev_ai = DevAIHelper()
        self.last_blob_thought_time = time.time()
        self.ai_queue = asyncio.Queue()
        self.max_queue_size = 3
        self.last_ai_call = time.time()
        self.min_ai_interval = 2.0
        self.ai_processor = None  # Will be initialized later
        self.state_file = "fish_state.json"
        self.load_state()  # Load saved state on startup
        self.last_save_time = time.time()
        self.save_interval = 1.0  # Save every second
        self.food_limit = 50  # Increased food limit
        self.food_rate_limit = 0.2  # Reduced cooldown to 200ms
        self.food_batch_queue = []
        self.last_food_process = time.time()
        self.min_food_process_interval = 0.016  # Process food more frequently (60fps)
        self.food_cooldown = {}  # Track food spawn cooldown per client
        self.max_fish = 50  # Maximum fish allowed
        self.min_fish = 10  # Minimum fish to maintain
        self.connected_clients = set()  # Change to store client IDs instead of WebSocket objects
        self.unique_ips = set()  # New: Track unique IPs
        self.think_batch_size = 1  # New: Process thoughts in batches
        self.update_interval = 0.033  # ~30fps for smooth animation
        self.broadcast_throttle = 0.033
        self.food_process_interval = 0.033
        self.state_cache_duration = 0.033
        self.high_load_threshold = 100  # Increased threshold
        self.food_limit = 40  # More food allowed
        self.food_rate_limit = 0.5  # Faster food spawning (0.5s cooldown)
        self.max_clients_per_ip = 5  # More lenient connection limit
        self.food_batch_size = 5  # Process more food at once
        self.min_food_process_interval = 0.033  # Process food more frequently
        self.broadcast_count = 0
        self.skip_frames = 0
        self.ip_connections = {}
        self.spawn_timer = 0
        self.spawn_interval = 2.0  # Spawn check every 2 seconds
        self.max_fish = 50
        self.min_fish = 15  # Increased minimum fish
        self.optimal_fish = 25  # Target fish count
        
        # Add missing time tracking attributes
        self.last_update = time.time()
        self.state_cache = None
        self.state_cache_time = time.time()
        
        # AI settings
        self.min_ai_interval = 2.0  # Faster AI updates
        self.max_queue_size = 3  # Allow more thoughts
        self.think_batch_size = 1  # Process one at a time for reliability
        self.last_ai_call = time.time()
        self.ai_queue = asyncio.Queue()
        self.ai_enabled = True  # New flag to track AI status
        self.last_blob_report_time = time.time()  # New: track last report time
        self.blob_report_interval = 30  # Generate report every 30 seconds
        self.blob_observation_interval = 5  # Quick observations every 5 seconds
        
        # Connection handling
        self.active_connections = set()
        self.connection_lock = asyncio.Lock()
        angle = random.uniform(0, 2 * math.pi)
        radius = 300  # Start shark away from center
        self.shark = Shark(
            self.width/2 + math.cos(angle) * radius,
            self.height/2 + math.sin(angle) * radius
        )
        self.shark_hunt_range = 300  # Increased range
        self.shark_eat_range = 30   # Increased eat range
        self.shark_hunt_cooldown = 2.0  # Shorter cooldown
        
        # Initialize squid with more prominent position and movement
        self.squid = Squid(
            x=self.width/2,
            y=self.height/2,
            vx=random.uniform(-3, 3),
            vy=random.uniform(-3, 3),
            size=6.0,  # Increased size
            color=[0.95, 0.6, 0.8]  # Bright pink
        )
        self.squid_dash_cooldown = 3.0  # More frequent dashes
        self.squid_dash_duration = 1.5  # Longer dashes
        self.squid_dash_timer = 0
        self.squid_ink_radius = 150  # Larger effect radius
        
        # Add fixed time step for movement
        self.fixed_time_step = 1/60  # 60 FPS target
        self.max_movement_scale = 1.0  # Cap movement scaling
        self.movement_speed_factor = 0.3  # Further reduce overall speed
        
        # Adjust entity speeds and behaviors
        self.fish_base_speed = 1.5  # Slower base speed
        self.shark_base_speed = 2.0  # Moderate shark speed
        self.squid_base_speed = 1.8  # Moderate squid speed
        self.food_detection_range = 150  # Range to detect food
        self.movement_speed_factor = 0.3  # Further reduce overall speed
        
        # Food-seeking parameters
        self.food_attraction_strength = 0.8
        self.food_energy_value = 30
        self.food_eat_distance = 15

    async def start_ai_processor(self):
        """Start the AI processor as a background task"""
        if self.ai_processor is None:
            self.ai_processor = asyncio.create_task(self._process_ai_queue())
        return self

    async def _process_ai_queue(self):
        """Fixed AI processing with improved blob thoughts"""
        while True:
            try:
                current_time = time.time()
                
                # Process regular AI queue
                if current_time - self.last_ai_call >= self.min_ai_interval:
                    if not self.ai_queue.empty():
                        request_type, entity, data = await self.ai_queue.get()
                        
                        try:
                            if request_type == 'blob':
                                fish_stats = [{
                                    'id': str(id(f)),
                                    'energy': f.energy,
                                    'speed': math.sqrt(f.vx * f.vx + f.vy * f.vy),
                                    'is_dying': f.energy < 30
                                } for f in self.fish]
                                
                                # Determine if it's time for a report
                                is_report_time = current_time - self.last_blob_report_time >= self.blob_report_interval
                                
                                if is_report_time:
                                    self.dev_ai.report_prompt  # Use report prompt
                                    self.last_blob_report_time = current_time
                                else:
                                    self.dev_ai.brief_prompt  # Use brief prompt
                                
                                thought = await self.dev_ai.get_response(fish_stats)
                                if thought:
                                    self.blob_fish.last_thought = thought
                                    logger.info(f"Blob thought: {thought}")
                            else:  # Fish thoughts
                                nearby_food = sum(1 for f in self.food 
                                                if math.sqrt((f['x'] - entity.x)**2 + 
                                                           (f['y'] - entity.y)**2) < 150)
                                
                                fish_state = {
                                    'energy': entity.energy,
                                    'vx': entity.vx,
                                    'vy': entity.vy,
                                    'nearby_food': nearby_food
                                }
                                
                                thought = await self.ai_helper.get_response(fish_state)
                                if thought:
                                    entity.last_thought = thought
                                    entity.think_timer = random.uniform(10, 20)  # Shorter think timer
                                    logger.info(f"Fish thought: {thought}")
                            
                            self.last_ai_call = current_time
                            
                        except Exception as e:
                            logger.error(f"Error in AI processing: {e}")
                            continue
                
                # Queue new thoughts
                if self.ai_queue.qsize() < self.max_queue_size:
                    current_time = time.time()
                    
                    # Queue blob thoughts
                    if current_time - self.last_blob_thought_time > self.blob_observation_interval:
                        await self.ai_queue.put(('blob', self.blob_fish, None))
                        self.last_blob_thought_time = current_time
                    
                    # Queue fish thoughts
                    thinking_candidates = [f for f in self.fish if f.think_timer <= 0]
                    if thinking_candidates:
                        fish = random.choice(thinking_candidates)
                        await self.ai_queue.put(('fish', fish, None))
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"AI queue error: {e}")
                await asyncio.sleep(1)

    async def save_state(self):
        """Save current simulation state to file"""
        state = {
            'fish': [{
                'x': f.x,
                'y': f.y,
                'vx': f.vx,
                'vy': f.vy,
                'energy': f.energy,
                'color': f.color,
                'size': f.size,
            } for f in self.fish],
            'food': self.food,
            'timestamp': datetime.now().isoformat(),
            'blob_fish': {
                'x': self.blob_fish.x,
                'y': self.blob_fish.y,
                'last_thought': self.blob_fish.last_thought
            }
        }
        
        try:
            async with aiofiles.open(self.state_file, 'w') as f:
                await f.write(json.dumps(state))
            logger.info("State saved successfully")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def load_state(self):
        """Load simulation state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.loads(f.read())
                
                # Recreate fish from saved state
                self.fish = [Fish(
                    x=f['x'],
                    y=f['y'],
                    vx=f['vx'],
                    vy=f['vy'],
                    energy=f['energy'],
                    color=f['color'],
                    size=f['size']
                ) for f in state['fish']]
                
                # Restore food
                self.food = state['food']
                
                # Restore blob fish position
                if 'blob_fish' in state:
                    self.blob_fish.x = state['blob_fish']['x']
                    self.blob_fish.y = state['blob_fish']['y']
                
                logger.info(f"State loaded from {state['timestamp']}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            # Initialize with default values if load fails
            self.fish = [Fish(random.uniform(0, self.width), random.uniform(0, self.height))
                        for _ in range(15)]
            self.food = []

    async def update(self):
        """Update with improved movement and food seeking"""
        current_time = time.time()
        elapsed = min(current_time - self.last_update, 0.1)
        self.last_update = current_time

        # Calculate movement scale
        movement_scale = min(self.fixed_time_step * 60, self.max_movement_scale) * self.movement_speed_factor
        
        # Update fish with food seeking
        for fish in self.fish:
            # Find nearest food
            nearest_food = None
            nearest_dist = float('inf')
            
            for food in self.food:
                dx = food['x'] - fish.x
                dy = food['y'] - fish.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                if dist < nearest_dist and dist < self.food_detection_range:
                    nearest_dist = dist
                    nearest_food = food
            
            # Move towards food if found, otherwise normal movement
            if nearest_food and fish.energy < 80:  # Only seek food if not full
                dx = nearest_food['x'] - fish.x
                dy = nearest_food['y'] - fish.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                # Eat food if close enough
                if dist < self.food_eat_distance:
                    fish.energy = min(100, fish.energy + self.food_energy_value)
                    self.food.remove(nearest_food)
                else:
                    # Move towards food with increased attraction
                    fish.vx += (dx/dist) * self.fish_base_speed * self.food_attraction_strength * movement_scale
                    fish.vy += (dy/dist) * self.fish_base_speed * self.food_attraction_strength * movement_scale
            else:
                # Normal wandering behavior
                dx = fish.target_x - fish.x
                dy = fish.target_y - fish.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                if dist < 20 or random.random() < 0.02:
                    fish.target_x = random.uniform(0, self.width)
                    fish.target_y = random.uniform(0, self.height)
                
                if dist > 0:
                    fish.vx += (dx/dist) * self.fish_base_speed * movement_scale * 0.5
                    fish.vy += (dy/dist) * self.fish_base_speed * movement_scale * 0.5

            # Apply velocity caps and updates
            speed = math.sqrt(fish.vx * fish.vx + fish.vy * fish.vy)
            if speed > self.fish_base_speed:
                fish.vx = (fish.vx/speed) * self.fish_base_speed
                fish.vy = (fish.vy/speed) * self.fish_base_speed

            # Update position with wrapping
            fish.x = (fish.x + fish.vx) % self.width
            fish.y = (fish.y + fish.vy) % self.height
            
            # Apply drag
            fish.vx *= 0.95
            fish.vy *= 0.95
            
            # Update energy and animation
            fish.energy = max(0, fish.energy - 0.05 * self.fixed_time_step)
            fish.phase += speed * 0.1
            fish.tail_angle = math.sin(fish.phase) * 0.3

        # Update shark with hunting behavior
        if self.shark:
            # Find nearest fish
            nearest_fish = None
            nearest_dist = float('inf')
            
            for fish in self.fish:
                dx = fish.x - self.shark.x
                dy = fish.y - self.shark.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                if dist < nearest_dist and dist < self.shark_hunt_range:
                    nearest_dist = dist
                    nearest_fish = fish
            
            if nearest_fish:
                # Hunt nearest fish
                dx = nearest_fish.x - self.shark.x
                dy = nearest_fish.y - self.shark.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                self.shark.vx += (dx/dist) * self.shark_base_speed * movement_scale
                self.shark.vy += (dy/dist) * self.shark_base_speed * movement_scale
            else:
                # Wander behavior
                dx = self.shark.target_x - self.shark.x
                dy = self.shark.target_y - self.shark.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                if dist < 30 or random.random() < 0.02:
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.uniform(100, 400)
                    self.shark.target_x = (self.width/2 + math.cos(angle) * radius) % self.width
                    self.shark.target_y = (self.height/2 + math.sin(angle) * radius) % self.height
                
                if dist > 0:
                    self.shark.vx += (dx/dist) * self.shark_base_speed * movement_scale * 0.5
                    self.shark.vy += (dy/dist) * self.shark_base_speed * movement_scale * 0.5
            
            # Update shark position and animation
            speed = math.sqrt(self.shark.vx * self.shark.vx + self.shark.vy * self.shark.vy)
            if speed > self.shark_base_speed:
                self.shark.vx = (self.shark.vx/speed) * self.shark_base_speed
                self.shark.vy = (self.shark.vy/speed) * self.shark_base_speed
            
            self.shark.x = (self.shark.x + self.shark.vx) % self.width
            self.shark.y = (self.shark.y + self.shark.vy) % self.height
            self.shark.vx *= 0.98
            self.shark.vy *= 0.98
            self.shark.phase += speed * 0.1
            self.shark.tail_angle = math.sin(self.shark.phase) * 0.2

        # Update squid with dynamic movement
        if self.squid:
            # Random dash initiation
            if self.squid_dash_timer <= 0 and random.random() < 0.01:
                self.squid_dash_timer = self.squid_dash_duration
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(200, 400)
                self.squid.target_x = (self.width/2 + math.cos(angle) * radius) % self.width
                self.squid.target_y = (self.height/2 + math.sin(angle) * radius) % self.height
            
            is_dashing = self.squid_dash_timer > 0
            self.squid_dash_timer = max(0, self.squid_dash_timer - elapsed)
            
            # Movement logic
            dx = self.squid.target_x - self.squid.x
            dy = self.squid.target_y - self.squid.y
            dist = math.sqrt(dx * dx + dy * dy)
            
            # Change target occasionally
            if dist < 30 or random.random() < 0.02:
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(100, 300)
                self.squid.target_x = (self.width/2 + math.cos(angle) * radius) % self.width
                self.squid.target_y = (self.height/2 + math.sin(angle) * radius) % self.height
            
            # Apply movement
            if dist > 0:
                acc = self.squid_base_speed * (2.0 if is_dashing else 1.0) * movement_scale
                self.squid.vx += (dx/dist) * acc
                self.squid.vy += (dy/dist) * acc
            
            # Cap speed
            speed = math.sqrt(self.squid.vx * self.squid.vx + self.squid.vy * self.squid.vy)
            max_speed = self.squid_base_speed * (2.0 if is_dashing else 1.0)
            if speed > max_speed:
                self.squid.vx = (self.squid.vx/speed) * max_speed
                self.squid.vy = (self.squid.vy/speed) * max_speed
            
            # Update position and animation
            self.squid.x = (self.squid.x + self.squid.vx) % self.width
            self.squid.y = (self.squid.y + self.squid.vy) % self.height
            self.squid.vx *= 0.97
            self.squid.vy *= 0.97
            self.squid.tentacle_phase += speed * 0.2

        # Adjust broadcast rate based on client count
        client_count = len(self.connected_clients)
        if client_count > 100:
            self.broadcast_throttle = 0.05  # 20 FPS for high load
        elif client_count > 50:
            self.broadcast_throttle = 0.033  # 30 FPS for medium load
        else:
            self.broadcast_throttle = 0.016  # 60 FPS for low load

        return []

    async def handle_ai_command(self, message):
        state = {
            'fish': [{
                'id': str(id(f)),
                'energy': f.energy,
                'speed': (f.vx**2 + f.vy**2)**0.5
            } for f in self.fish]
        }
        return await self.ai_helper.get_response(state)

    async def get_state(self):
        """Cached state retrieval with viewer count logic"""
        current_time = time.time()
        if (self.state_cache and 
            current_time - self.state_cache_time < self.state_cache_duration):
            return self.state_cache

        viewer_count = len(self.unique_ips)
        
        state = {
            'fish': [{
                'id': str(id(f)),
                'x': f.x,
                'y': f.y,
                'vx': f.vx,
                'vy': f.vy,
                'size': f.size,
                'energy': f.energy,
                'color': f.color,
                'tail_angle': f.tail_angle,
                'thought': f.last_thought
            } for f in self.fish],
            'food': self.food,
            'dimensions': {
                'width': self.width,
                'height': self.height
            },
            'viewer_count': viewer_count if viewer_count >= 5 else 0,
            'shark': {
                'x': self.shark.x,
                'y': self.shark.y,
                'vx': self.shark.vx,
                'vy': self.shark.vy,
                'tail_angle': self.shark.tail_angle,
                'energy': self.shark.energy,
                'color': self.shark.color,
                'size': self.shark.size
            },
            'squid': {
                'x': self.squid.x,
                'y': self.squid.y,
                'vx': self.squid.vx,
                'vy': self.squid.vy,
                'size': self.squid.size,
                'color': self.squid.color,
                'tentacle_phase': self.squid.tentacle_phase,
                'is_dashing': self.squid_dash_timer > 0
            } if self.squid else None
        }
        
        if self.blob_fish:
            state['blob_fish'] = {
                'x': self.blob_fish.x,
                'y': self.blob_fish.y,
                'tail_angle': self.blob_fish.tail_angle,
                'thought': self.blob_fish.last_thought
            }

        self.state_cache = state
        self.state_cache_time = current_time
        return state

    async def add_food(self, x, y, client_id, client_ip):
        """Simplified food addition"""
        current_time = time.time()
        
        # Basic rate limiting
        if client_id in self.food_cooldown:
            if current_time - self.food_cooldown[client_id] < self.food_rate_limit:
                return False
        
        # Directly add food instead of using queue
        if len(self.food) < self.food_limit:
            self.food.append({
                'x': x,
                'y': y,
                'energy': 30
            })
            self.food_cooldown[client_id] = current_time
            return True
        return False

    async def process_food_queue(self):
        """Smoother food processing"""
        current_time = time.time()
        if current_time - self.last_food_process < self.min_food_process_interval:
            return

        if not self.food_batch_queue:
            return

        # Process food in smaller batches more frequently
        items_to_process = min(len(self.food_batch_queue), self.food_batch_size)
        for _ in range(items_to_process):
            if self.food_batch_queue and len(self.food) < self.food_limit:
                food_item = self.food_batch_queue.pop(0)
                if current_time - food_item['timestamp'] < 2.0:  # More lenient timestamp check
                    self.food.append(food_item)

        self.last_food_process = current_time

    def add_client(self, websocket, client_ip):  # Modified to accept IP
        self.connected_clients.add(str(id(websocket)))
        if client_ip:  # Only add valid IPs
            self.unique_ips.add(client_ip)
        
    def remove_client(self, websocket, client_ip):  # Modified to accept IP
        client_id = str(id(websocket))
        self.connected_clients.discard(client_id)
        if client_ip:
            self.unique_ips.discard(client_ip)

    async def cleanup_connection(self, websocket, client_id, client_ip):
        """Safely cleanup disconnected client"""
        async with self.connection_lock:
            if client_ip in self.ip_connections:
                self.ip_connections[client_ip].discard(client_id)
                if not self.ip_connections[client_ip]:
                    del self.ip_connections[client_ip]
            
            self.connected_clients.discard(client_id)
            self.unique_ips.discard(client_ip)
            self.active_connections.discard(websocket)
            
            if client_id in self.food_cooldown:
                del self.food_cooldown[client_id]

    async def add_connection(self, websocket, client_id, client_ip):
        """Safely add new connection"""
        async with self.connection_lock:
            self.active_connections.add(websocket)
            self.connected_clients.add(client_id)
            if client_ip:
                self.unique_ips.add(client_ip)
                if client_ip in self.ip_connections:
                    self.ip_connections[client_ip].add(client_id)
                else:
                    self.ip_connections[client_ip] = {client_id}


app = FastAPI()

# Serve static files from a 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend files not found")

# Initialize sim with event loop
@app.on_event("startup")
async def startup_event():
    global sim
    sim = FishSim()
    await sim.start_ai_processor()
    # Load initial state
    sim.load_state()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        if hasattr(sim, 'ai_processor') and sim.ai_processor:
            sim.ai_processor.cancel()
        # Save final state
        await sim.save_state()
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(id(websocket))
    client_ip = websocket.client.host
    if websocket.headers.get('x-forwarded-for'):
        client_ip = websocket.headers.get('x-forwarded-for').split(',')[0].strip()
    
    try:
        # Check connection limits before accepting
        if client_ip in sim.ip_connections:
            if len(sim.ip_connections[client_ip]) >= sim.max_clients_per_ip:
                await websocket.close(code=1000)
                return
        
        await websocket.accept()
        await sim.add_connection(websocket, client_id, client_ip)
        
        async def broadcast_loop():
            last_broadcast = 0
            try:
                while True:
                    current_time = time.time()
                    if current_time - last_broadcast >= sim.broadcast_throttle:
                        if websocket not in sim.active_connections:
                            break
                            
                        await sim.update()
                        state = await sim.get_state()
                        await websocket.send_json(state)
                        last_broadcast = current_time
                        
                        # Smoother adaptive sleep
                        client_count = len(sim.connected_clients)
                        sleep_time = (
                            0.05 if client_count > 200 else
                            0.033 if client_count > 100 else
                            0.016
                        )
                        await asyncio.sleep(sleep_time)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"Broadcast error: {str(e)}")

        async def receive_loop():
            try:
                while True:
                    if websocket not in sim.active_connections:
                        break
                        
                    data = await websocket.receive_json()
                    current_time = time.time()
                    
                    if data['type'] == 'add_food':
                        await sim.add_food(data['x'], data['y'], client_id, client_ip)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"Receive error: {str(e)}")

        # Run both loops concurrently
        await asyncio.gather(
            broadcast_loop(),
            receive_loop(),
            return_exceptions=True
        )
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up safely
        await sim.cleanup_connection(websocket, client_id, client_ip)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True,
        ssl_keyfile=os.environ.get("SSL_KEYFILE"),
        ssl_certfile=os.environ.get("SSL_CERTFILE"),
        ws_max_size=1024*1024  # 1MB max message size
    )