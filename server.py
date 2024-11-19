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

    def __post_init__(self):
        if self.color is None:
            hue = random.uniform(180, 250)
            sat = random.uniform(0.7, 0.9)
            val = random.uniform(0.7, 0.9)
            self.color = self._hsv_to_rgb(hue, sat, val)
            self.target_x = random.uniform(0, 1200)
            self.target_y = random.uniform(0, 800)
            self.phase = random.uniform(0, 2 * math.pi)

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


class FishSim:
    def __init__(self):
        self.width = 1200
        self.height = 800
        self.fish = [Fish(random.uniform(0, self.width), random.uniform(0, self.height))
                     for _ in range(15)]
        self.food = []
        self.start_time = time.time()

    async def update(self):
        current_time = time.time() - self.start_time
        code_updates = []

        for fish in self.fish:
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

        return code_updates


app = FastAPI()

# Serve static files from a 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend files not found")

sim = FishSim()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def simulation_loop():
        while True:
            code_updates = await sim.update()
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
                    'code': f.last_code
                } for f in sim.fish],
                'food': sim.food,
                'dimensions': {
                    'width': sim.width,
                    'height': sim.height
                }
            }
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

    await asyncio.gather(simulation_loop(), receive_loop())