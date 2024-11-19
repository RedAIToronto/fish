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


class AIHelper:
    def __init__(self):
        self.client = anthropic.Anthropic()
        logger.info("AIHelper initialized with Anthropic client")
        self.system_prompt = """You are a fish in an aquarium. Respond with ONLY 2-3 WORDS and an emoji that perfectly matches the current state.
        
        Consider these states carefully:
        - If energy < 30: Show distress about dying/hunger
        - If energy > 70: Show happiness/excitement
        - If nearby_food > 0: React to food
        - If speed > 4: Express excitement about speed
        - If speed < 1: Express tiredness/laziness
        
        Example responses for different states:
        Low energy: "💀 dying... help!" or "😫 need food badly"
        High energy: "😊 feeling great!" or "🌟 so energetic"
        Near food: "🍽️ food spotted!" or "😋 yummy ahead"
        High speed: " zoom zoom!" or "💨 super fast"
        Low speed: "😴 so sleepy" or "🦥 just floating"
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
        self.report_interval = 30  # Generate report every 30 seconds
        self.brief_prompt = """You are a cute dev blob fish (with glasses) studying your aquarium.
        KEEP RESPONSES UNDER 50 CHARS! Focus on immediate observations.
        
        Examples:
        "🤓 *notes* Fish #12 swimming erratically"
        "📊 Observing feeding patterns..."
        "🔬 Studying group dynamics"
        "🧪 Monitoring energy levels"
        """
        
        self.report_prompt = """You are a scientist blob fish writing a brief research report about your aquarium ecosystem.
        Write ONE short paragraph (max 200 chars) analyzing:
        - Fish behavior patterns
        - Population dynamics
        - Energy distribution
        - Social interactions
        - Ecosystem stability
        
        Use academic but cute tone. Include emoji. Sign as "Dr. Blob".
        
        Example:
        "🔬 Field Report: Observed fascinating schooling behavior among specimens. Energy levels show cyclical patterns 
        correlating with feeding events. Population stable at optimal levels. Social dynamics indicate emergent intelligence.
        
        - Dr. Blob 🎓"
        """

    async def get_response(self, fish_stats):
        current_time = time.time()
        current_stats = {
            'population': len(fish_stats),
            'dying_count': sum(1 for f in fish_stats if f['energy'] < 30),
            'active_count': sum(1 for f in fish_stats if f['speed'] > 3),
            'avg_energy': sum(f['energy'] for f in fish_stats)/len(fish_stats)
        }

        # Decide if it's time for a report
        is_report_time = current_time - self.last_report_time > self.report_interval

        try:
            if is_report_time:
                # Generate detailed report
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.messages.create(
    model="claude-3-5-sonnet-20241022",
                        max_tokens=400,
                        temperature=0.9,
                        system=self.report_prompt,
                        messages=[{
                            "role": "user",
                            "content": f"Current ecosystem state: {current_stats['population']} specimens, {current_stats['dying_count']} critical, {current_stats['active_count']} active, average energy {current_stats['avg_energy']:.1f}"
                        }]
                    )
                )
                self.last_report_time = current_time
            else:
                # Generate quick observation
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.messages.create(
    model="claude-3-5-sonnet-20241022",
                        max_tokens=80,
                        temperature=0.9,
                        system=self.brief_prompt,
                        messages=[{
                            "role": "user",
                            "content": f"Observing: {current_stats['dying_count']} critical fish, {current_stats['active_count']} active"
                        }]
                    )
                )

            thought = str(response.content[0].text if isinstance(response.content, list) else response.content)
            return thought[:200 if is_report_time else 50]  # Limit length based on type
            
        except Exception as e:
            logger.error(f"Dev blob fish error: {str(e)}")
            if current_stats['dying_count'] > 0:
                return "🤓 *concerned* Monitoring critical specimens..."
            return "🔬 *adjusts glasses* Continuing observations..."


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
        self.max_queue_size = 2
        self.last_ai_call = 0
        self.min_ai_interval = 3.0
        self.ai_processor = None  # Will be initialized later
        self.state_file = "fish_state.json"
        self.load_state()  # Load saved state on startup
        self.last_save_time = time.time()
        self.save_interval = 30  # Save every 30 seconds
        self.food_limit = 30  # Reduce max food items
        self.food_rate_limit = 2.0  # Increase cooldown between food spawns
        self.food_batch_queue = []  # New: Queue for batching food additions
        self.last_food_process = time.time()
        self.food_process_interval = 0.1  # Process food every 100ms
        self.food_cooldown = {}  # Track food spawn cooldown per client
        self.max_fish = 50  # Maximum fish allowed
        self.min_fish = 10  # Minimum fish to maintain
        self.connected_clients = set()  # Change to store client IDs instead of WebSocket objects
        self.unique_ips = set()  # New: Track unique IPs
        self.think_batch_size = 3  # New: Process thoughts in batches
        self.update_interval = 0.1  # Reduce update frequency
        self.broadcast_throttle = 0.1  # Throttle broadcasts
        self.last_broadcast = time.time()
        self.max_clients_per_ip = 3  # Limit connections per IP
        self.ip_connections = {}  # Track connections per IP
        self.last_update = time.time()
        self.broadcast_queue = asyncio.Queue()  # New: Queue for broadcasting state
        self.state_cache = None  # New: Cache state between broadcasts
        self.state_cache_time = 0
        self.state_cache_duration = 0.1  # Cache state for 100ms
        self.high_load_mode = False
        self.load_check_interval = 5.0  # Check load every 5 seconds
        self.last_load_check = time.time()
        
    async def start_ai_processor(self):
        """Start the AI processor as a background task"""
        if self.ai_processor is None:
            self.ai_processor = asyncio.create_task(self._process_ai_queue())
        return self

    async def _process_ai_queue(self):
        """Optimized AI request processing"""
        while True:
            try:
                if self.ai_queue.qsize() > 0:
                    current_time = time.time()
                    if current_time - self.last_ai_call >= self.min_ai_interval:
                        # Process multiple thoughts at once
                        thoughts_to_process = []
                        for _ in range(self.think_batch_size):
                            if self.ai_queue.qsize() > 0:
                                thoughts_to_process.append(await self.ai_queue.get())
                            else:
                                break

                        # Batch process thoughts
                        for request_type, entity, data in thoughts_to_process:
                            try:
                                if request_type == 'blob':
                                    thought = await self.dev_ai.get_response(data)
                                    self.blob_fish.last_thought = thought
                                else:
                                    nearby_food = sum(1 for f in self.food 
                                                    if np.hypot(f['x'] - entity.x, f['y'] - entity.y) < 150)
                                    fish_state = {
                                        'energy': entity.energy,
                                        'vx': entity.vx,
                                        'vy': entity.vy,
                                        'nearby_food': nearby_food
                                    }
                                    thought = await self.ai_helper.get_response(fish_state)
                                    entity.last_thought = thought
                                    entity.think_timer = random.uniform(30, 60)  # Increase think timer
                            except Exception as e:
                                logger.error(f"Error processing thought: {e}")
                                continue

                        self.last_ai_call = current_time
                
                await asyncio.sleep(0.5)  # Increased sleep time
            except Exception as e:
                logger.error(f"AI queue processing error: {e}")
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
        """Optimized update loop"""
        current_time = time.time()
        
        # Check system load
        if current_time - self.last_load_check > self.load_check_interval:
            self.high_load_mode = len(self.connected_clients) > 50
            self.last_load_check = current_time
        
        # Process food queue
        await self.process_food_queue()
        
        # Adjust simulation based on load
        if self.high_load_mode:
            self.update_interval = 0.2  # Reduce update frequency under high load
            self.food_limit = 20  # Reduce food limit
            self.think_batch_size = 2  # Reduce AI processing
        else:
            self.update_interval = 0.1
            self.food_limit = 30
            self.think_batch_size = 3
        
        # Save state periodically
        if current_time - self.last_save_time > self.save_interval:
            await self.save_state()
            self.last_save_time = current_time
        
        # Queue blob fish thoughts less frequently
        if (current_time - self.last_blob_thought_time > 8 and 
            self.ai_queue.qsize() < self.max_queue_size):
            
            fish_stats = [{
                'id': str(id(f)),
                'energy': f.energy,
                'speed': (f.vx**2 + f.vy**2)**0.5,
                'is_dying': f.energy < 30
            } for f in self.fish]
            
            await self.ai_queue.put(('blob', self.blob_fish, fish_stats))
            self.last_blob_thought_time = current_time

        # Queue regular fish thoughts
        thinking_candidates = [f for f in self.fish if f.think_timer <= 0]
        if thinking_candidates and self.ai_queue.qsize() < self.max_queue_size:
            fish = random.choice(thinking_candidates)
            await self.ai_queue.put(('fish', fish, None))
            fish.think_timer = 5  # Short timer in case AI fails

        # Regular simulation updates
        code_updates = []
        
        for fish in self.fish:
            fish.think_timer -= 0.05
            
            # Natural swimming behavior
            dx = fish.target_x - fish.x
            dy = fish.target_y - fish.y
            dist = np.hypot(dx, dy)

            if dist < 20 or random.random() < 0.01:
                fish.target_x = random.uniform(0, self.width)
                fish.target_y = random.uniform(0, self.height)

            if dist > 0:
                fish.vx += (dx / dist) * 0.2
                fish.vy += (dy / dist) * 0.2

            if self.food:
                food_dist = [(f, np.hypot(f['x'] - fish.x, f['y'] - fish.y))
                             for f in self.food]
                nearest = min(food_dist, key=lambda x: x[1])
                if nearest[1] < 150:
                    dx = nearest[0]['x'] - fish.x
                    dy = nearest[0]['y'] - fish.y
                    dist = np.hypot(dx, dy)
                    if dist > 0:
                        fish.vx += (dx / dist) * 0.8
                        fish.vy += (dy / dist) * 0.8
                    if dist < 10:
                        fish.energy += 30
                        self.food.remove(nearest[0])

            speed = np.hypot(fish.vx, fish.vy)
            max_speed = 5
            if speed > max_speed:
                fish.vx = (fish.vx / speed) * max_speed
                fish.vy = (fish.vy / speed) * max_speed

            fish.phase += speed * 0.1
            fish.tail_angle = math.sin(fish.phase) * (0.2 + min(speed / max_speed, 1) * 0.3)

            code_updates.append(fish.update_behavior(dx, dy, dist, speed))

            fish.x = (fish.x + fish.vx) % self.width
            fish.y = (fish.y + fish.vy) % self.height

            fish.vx *= 0.98
            fish.vy *= 0.98

            fish.energy -= 0.1

        self.fish = [f for f in self.fish if f.energy > 0]
        if random.random() < 0.02:
            self.fish.append(Fish(random.uniform(0, self.width),
                                  random.uniform(0, self.height)))

        # Gentle floating movement for blob fish
        self.blob_fish.x += math.sin(time.time() * 0.5) * 0.3
        self.blob_fish.y += math.cos(time.time() * 0.3) * 0.2
        self.blob_fish.tail_angle = math.sin(time.time()) * 0.1

        # Population control
        if len(self.fish) > self.max_fish:
            # Remove excess fish, keeping the healthiest ones
            self.fish.sort(key=lambda f: f.energy, reverse=True)
            self.fish = self.fish[:self.max_fish]
        elif len(self.fish) < self.min_fish:
            # Spawn new fish up to minimum
            for _ in range(self.min_fish - len(self.fish)):
                self.fish.append(Fish(
                    random.uniform(0, self.width),
                    random.uniform(0, self.height)
                ))

        # Cleanup old food
        if len(self.food) > self.food_limit:
            self.food = self.food[-self.food_limit:]

        # Clean up old cooldowns
        self.food_cooldown = {
            client: time 
            for client, time in self.food_cooldown.items() 
            if current_time - time < 60  # Remove cooldowns older than 1 minute
        }

        return code_updates

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

        viewer_count = len(self.unique_ips)  # Use unique IPs instead of connections
        
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
            'viewer_count': viewer_count if viewer_count >= 5 else 0  # Only show if ≥ 5 viewers
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
        """Optimized food addition with batching"""
        current_time = time.time()
        
        # Enhanced rate limiting
        if client_id in self.food_cooldown:
            if current_time - self.food_cooldown[client_id] < self.food_rate_limit:
                return False
                
        # IP-based rate limiting
        if client_ip in self.ip_connections:
            if len(self.ip_connections[client_ip]) >= self.max_clients_per_ip:
                return False
        
        # Add to batch queue instead of immediate processing
        self.food_batch_queue.append({
            'x': x,
            'y': y,
            'energy': 30,
            'timestamp': current_time
        })
        
        self.food_cooldown[client_id] = current_time
        return True

    async def process_food_queue(self):
        """Process food additions in batches"""
        current_time = time.time()
        if current_time - self.last_food_process < self.food_process_interval:
            return

        if not self.food_batch_queue:
            return

        # Process only the most recent food items if we have too many
        if len(self.food_batch_queue) > 10:
            self.food_batch_queue = self.food_batch_queue[-10:]

        # Add food items from queue
        while self.food_batch_queue and len(self.food) < self.food_limit:
            food_item = self.food_batch_queue.pop(0)
            if current_time - food_item['timestamp'] < 5.0:  # Only add recent food
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

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(sim, 'ai_processor') and sim.ai_processor:
        sim.ai_processor.cancel()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(id(websocket))
    
    # Get client IP and check connection limits
    client_ip = websocket.client.host
    if websocket.headers.get('x-forwarded-for'):
        client_ip = websocket.headers.get('x-forwarded-for').split(',')[0].strip()
    
    # Check IP connection limit
    if client_ip in sim.ip_connections:
        if len(sim.ip_connections[client_ip]) >= sim.max_clients_per_ip:
            await websocket.close(code=1008, reason="Too many connections")
            return
        sim.ip_connections[client_ip].add(client_id)
    else:
        sim.ip_connections[client_ip] = {client_id}
    
    await websocket.accept()
    sim.add_client(websocket, client_ip)
    
    try:
        async def broadcast_loop():
            last_broadcast = 0
            while True:
                try:
                    current_time = time.time()
                    if current_time - last_broadcast >= sim.broadcast_throttle:
                        state = await sim.get_state()
                        await websocket.send_json(state)
                        last_broadcast = current_time
                        
                        # Adaptive sleep based on load
                        sleep_time = 0.1 if sim.high_load_mode else 0.05
                        await asyncio.sleep(sleep_time)
                    else:
                        await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")
                    break

        async def receive_loop():
            while True:
                try:
                    data = await websocket.receive_json()
                    if data['type'] == 'add_food':
                        await sim.add_food(
                            data['x'],
                            data['y'],
                            client_id,
                            client_ip
                        )
                    await asyncio.sleep(0.05)  # Add small delay between messages
                except Exception as e:
                    logger.error(f"Receive error: {e}")
                    break

        await asyncio.gather(broadcast_loop(), receive_loop())
    finally:
        # Cleanup
        if client_ip in sim.ip_connections:
            sim.ip_connections[client_ip].discard(client_id)
            if not sim.ip_connections[client_ip]:
                del sim.ip_connections[client_ip]
        sim.remove_client(websocket, client_ip)
        if client_id in sim.food_cooldown:
            del sim.food_cooldown[client_id]

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