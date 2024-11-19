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
                        max_tokens=200,
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
                        max_tokens=50,
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
        self.max_queue_size = 3
        self.last_ai_call = 0
        self.min_ai_interval = 1.0
        self.ai_processor = None  # Will be initialized later

    async def start_ai_processor(self):
        """Start the AI processor as a background task"""
        if self.ai_processor is None:
            self.ai_processor = asyncio.create_task(self._process_ai_queue())
        return self

    async def _process_ai_queue(self):
        """Background task to process AI requests"""
        while True:
            try:
                if self.ai_queue.qsize() > 0:
                    current_time = time.time()
                    if current_time - self.last_ai_call >= self.min_ai_interval:
                        request_type, entity, data = await self.ai_queue.get()
                        
                        if request_type == 'blob':
                            thought = await self.dev_ai.get_response(data)
                            self.blob_fish.last_thought = thought
                            logger.info(f"Blob thought: {thought}")
                        else:
                            nearby_food = sum(1 for f in self.food 
                                            if np.hypot(f['x'] - entity.x, f['y'] - entity.y) < 150)
                            
                            fish_state = {
                                'energy': entity.energy,
                                'vx': entity.vx,
                                'vy': entity.vy,
                                'x': entity.x,
                                'y': entity.y,
                                'nearby_food': nearby_food
                            }
                            
                            thought = await self.ai_helper.get_response(fish_state)
                            entity.last_thought = thought
                            entity.think_timer = random.uniform(20, 40)
                        
                        self.last_ai_call = current_time
                
                await asyncio.sleep(0.1)  # Prevent CPU hogging
            except Exception as e:
                logger.error(f"AI queue processing error: {e}")
                await asyncio.sleep(1)  # Wait longer on error

    async def update(self):
        current_time = time.time()
        
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
            }
        }
        state['blob_fish'] = {
            'x': self.blob_fish.x,
            'y': self.blob_fish.y,
            'tail_angle': self.blob_fish.tail_angle,
            'thought': self.blob_fish.last_thought
        }
        return state


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
    await websocket.accept()

    async def simulation_loop():
        while True:
            code_updates = await sim.update()
            state = await sim.get_state()
            await websocket.send_json(state)
            await asyncio.sleep(0.05)

    async def receive_loop():
        while True:
            data = await websocket.receive_json()
            if data['type'] == 'add_food':
                sim.food.append({
                    'x': data['x'],
                    'y': data['y'],
                    'energy': 30
                })
            elif data['type'] == 'ai_query':
                response = await sim.handle_ai_command(data['message'])
                await websocket.send_json({
                    'type': 'ai_response',
                    'message': response
                })

    await asyncio.gather(simulation_loop(), receive_loop())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)