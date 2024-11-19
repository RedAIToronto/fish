const canvas = document.getElementById('fishTank');
const ctx = canvas.getContext('2d');
const thoughtLog = document.getElementById('thoughtLog');
const MAX_THOUGHTS = 3;
const SPAWN_EFFECTS = new Map();  // Store spawn effects
const DEATH_EFFECTS = new Map();  // Store death effects
const EFFECT_DURATION = 1000;  // Effect duration in ms

// Initialize variables for tracking fish
let previousFishIds = new Set();
let thoughtCache = new Map();
const THOUGHT_CACHE_DURATION = 5000;

// WebSocket connection handling
let ws;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 1000;

// Add constants at the top
const FPS = 60;
const frameDelay = 1000 / FPS;
let lastFrameTime = 0;
let viewerCount = 0;

// Initialize WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    console.log('Connecting to WebSocket:', wsUrl);
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
    };
    
    ws.onclose = (e) => {
        console.log('WebSocket closed:', e.code, e.reason);
        if (reconnectAttempts < maxReconnectAttempts) {
            setTimeout(() => {
                reconnectAttempts++;
                connectWebSocket();
            }, reconnectDelay * Math.pow(2, reconnectAttempts)); // Exponential backoff
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onmessage = async function(event) {
        try {
            const state = JSON.parse(event.data);
            
            // Update viewer count if provided
            if (state.viewer_count !== undefined) {
                viewerCount = state.viewer_count;
                updateViewerCount(viewerCount);
            }

            // Frame rate control
            const currentTime = Date.now();
            const elapsed = currentTime - lastFrameTime;
            
            if (elapsed < frameDelay) {
                return; // Skip frame if too soon
            }
            lastFrameTime = currentTime;

            console.log('Received state:', {
                fishCount: state.fish.length,
                foodCount: state.food.length,
                blobFish: !!state.blob_fish
            });
            
            // Track fish spawns and deaths
            const currentFishIds = new Set(state.fish.map(f => f.id));
            
            // Check for new fish (spawns)
            for (const fish of state.fish) {
                if (!previousFishIds.has(fish.id)) {
                    SPAWN_EFFECTS.set(fish.id, {
                        x: fish.x,
                        y: fish.y,
                        startTime: Date.now()
                    });
                }
            }
            
            // Check for removed fish (deaths)
            for (const oldId of previousFishIds) {
                if (!currentFishIds.has(oldId)) {
                    // Find the last known position of the fish
                    const deadFish = state.fish.find(f => f.id === oldId) || 
                                   Array.from(previousFishIds).find(f => f.id === oldId);
                    if (deadFish) {
                        DEATH_EFFECTS.set(oldId, {
                            x: deadFish.x,
                            y: deadFish.y,
                            startTime: Date.now()
                        });
                    }
                }
            }
            
            previousFishIds = currentFishIds;
            
            // Clear canvas and draw everything
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw ambient particles
            for(let i = 0; i < 50; i++) {
                ctx.fillStyle = 'rgba(255,255,255,0.1)';
                ctx.beginPath();
                ctx.arc(
                    (Date.now()/20 + i * 100) % canvas.width,
                    (Math.sin((Date.now() + i * 1000)/1000) * 50) + i * 16,
                    1,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
            }
            
            // Draw food
            state.food.forEach(food => {
                ctx.save();
                ctx.shadowColor = '#4ADE80';
                ctx.shadowBlur = 10;
                ctx.fillStyle = '#4ADE80';
                ctx.beginPath();
                ctx.arc(food.x, food.y, 3, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            });
            
            // Draw fish
            state.fish.forEach(fish => {
                drawFish(fish);
                drawFishThought(ctx, fish);
                if (fish.thought) {
                    addThought(fish);
                }
            });
            
            // Draw effects
            drawEffects(ctx);
            
            // Update data panel
            updateDataPanel(state.fish, state.food.length);
            
            // Draw blob fish
            if (state.blob_fish) {
                drawBlobFish(state.blob_fish);
                drawDevThought(ctx, state.blob_fish);
                updateDevPanel(state.blob_fish.thought);
            }
            
        } catch (error) {
            console.error('Error handling message:', error);
        }
    };
}

// Start the connection when the page loads
connectWebSocket();

function drawFish(fish) {
    ctx.save();
    ctx.translate(fish.x, fish.y);

    const direction = fish.vx >= 0 ? 1 : -1;
    ctx.scale(direction, 1);

    // Calculate energy-based opacity and color fade
    const energyRatio = fish.energy / 100;
    const fadeToGray = 1 - energyRatio;
    
    // Shadow for depth
    ctx.shadowColor = 'rgba(0,0,0,0.2)';
    ctx.shadowBlur = 10;

    // Body gradient with energy-based color
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 25);
    const baseColor = fish.color.map(c => c * 255);
    const fadedColor = baseColor.map(c => c * energyRatio + (200 * fadeToGray)); // Fade to light gray
    
    gradient.addColorStop(0, `rgba(${fadedColor.join(',')}, 1)`);
    gradient.addColorStop(1, `rgba(${fadedColor.map(c => c * 0.7).join(',')}, 1)`);
    ctx.fillStyle = gradient;

    // Main body
    ctx.beginPath();
    ctx.moveTo(-20, 0);
    ctx.quadraticCurveTo(0, -15, 20, 0);
    ctx.quadraticCurveTo(0, 15, -20, 0);
    ctx.fill();

    // Tail
    ctx.save();
    ctx.translate(-20, 0);
    ctx.rotate(fish.tail_angle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(-15, -10, -20, 0);
    ctx.quadraticCurveTo(-15, 10, 0, 0);
    ctx.fill();
    ctx.restore();

    // Dorsal fin
    ctx.beginPath();
    ctx.moveTo(0, -15);
    ctx.quadraticCurveTo(5, -20, 10, -15);
    ctx.quadraticCurveTo(5, -10, 0, -15);
    ctx.fill();

    // Eye
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(10, -5, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(11, -5, 1.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
}

function drawFishThought(ctx, fish) {
    if (!fish.thought) return;
    
    ctx.save();
    
    // Smaller bubble position
    const bubbleX = fish.x;
    const bubbleY = fish.y - 30;  // Closer to fish
    
    // Setup text style
    ctx.font = '12px Arial';  // Smaller font
    const metrics = ctx.measureText(fish.thought);
    const padding = 8;
    
    // Smaller bubble dimensions
    const bubbleWidth = metrics.width + padding * 2;
    const bubbleHeight = 20;  // Fixed small height
    
    // Simple white bubble with slight transparency
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    
    // Minimal bubble
    ctx.beginPath();
    ctx.roundRect(
        bubbleX - bubbleWidth/2,
        bubbleY - bubbleHeight/2,
        bubbleWidth,
        bubbleHeight,
        5
    );
    ctx.fill();
    
    // Simple tail
    ctx.beginPath();
    ctx.moveTo(bubbleX - 5, bubbleY + bubbleHeight/2);
    ctx.lineTo(bubbleX, bubbleY + bubbleHeight/2 + 5);
    ctx.lineTo(bubbleX + 5, bubbleY + bubbleHeight/2);
    ctx.fill();
    
    // Draw text
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(fish.thought, bubbleX, bubbleY);
    
    ctx.restore();
}

function updateDataPanel(fish, foodCount) {
    const globalStats = document.getElementById('globalStats');
    const fishData = document.getElementById('fishData');
    const fishCount = document.getElementById('fishCount');

    // Update global stats
    globalStats.innerHTML = `
        Population: ${fish.length}<br>
        Avg Energy: ${(fish.reduce((sum, f) => sum + f.energy, 0) / fish.length).toFixed(2)}<br>
        Active Food: ${foodCount}
    `;

    fishCount.innerHTML = `Supply: ${fish.length}`;

    // Update individual fish data
    fishData.innerHTML = fish.map(f => {
        const speed = Math.sqrt(f.vx * f.vx + f.vy * f.vy);
        return `
            <div class="fish-data" style="border-left: 4px solid rgb(${f.color.map(c => c * 255).join(',')})">
                Fish #${f.id.slice(-4)}<br>
                Energy: ${f.energy.toFixed(1)}<br>
                Speed: ${speed.toFixed(2)}
                ${createVectorDisplay(f.vx, f.vy)}
            </div>
        `;
    }).join('');
}

function createVectorDisplay(vx, vy) {
    const magnitude = Math.sqrt(vx * vx + vy * vy);
    const angle = Math.atan2(vy, vx);
    const length = Math.min(magnitude * 10, 25);

    return `
        <div class="vector-display">
            <div class="vector-arrow" style="
                width: ${length}px;
                transform: rotate(${angle}rad);
                background: rgb(${Math.min(255, magnitude * 50)},
                             ${Math.max(0, 255 - magnitude * 50)},
                             255);
            "></div>
        </div>
    `;
}

function addThought(fish) {
    if (!fish.thought) return;
    
    const currentTime = Date.now();
    
    // Clean old thoughts from cache
    for (const [key, time] of thoughtCache) {
        if (currentTime - time > THOUGHT_CACHE_DURATION) {
            thoughtCache.delete(key);
        }
    }
    
    // Create unique key for this thought
    const thoughtKey = `${fish.id}-${fish.thought}`;
    
    // Check if this thought was recently displayed
    if (thoughtCache.has(thoughtKey)) {
        return;
    }
    
    // Add to cache
    thoughtCache.set(thoughtKey, currentTime);
    
    const thoughtEntry = document.createElement('div');
    thoughtEntry.className = 'thought-entry';
    
    const time = new Date().toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit'
    });
    
    thoughtEntry.textContent = `[${time}] ${fish.thought}`;
    
    thoughtLog.insertBefore(thoughtEntry, thoughtLog.firstChild);
    
    while (thoughtLog.children.length > MAX_THOUGHTS) {
        thoughtLog.removeChild(thoughtLog.lastChild);
    }
}

function drawEffects(ctx) {
    const currentTime = Date.now();
    
    // Draw spawn effects
    for (const [id, effect] of SPAWN_EFFECTS.entries()) {
        const progress = (currentTime - effect.startTime) / EFFECT_DURATION;
        if (progress >= 1) {
            SPAWN_EFFECTS.delete(id);
            continue;
        }
        
        // Expanding circle effect
        ctx.save();
        ctx.strokeStyle = `rgba(74, 222, 128, ${1 - progress})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(effect.x, effect.y, progress * 40, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }
    
    // Enhanced death effects
    for (const [id, effect] of DEATH_EFFECTS.entries()) {
        const progress = (currentTime - effect.startTime) / EFFECT_DURATION;
        if (progress >= 1) {
            DEATH_EFFECTS.delete(id);
            continue;
        }
        
        ctx.save();
        
        // Large fading red circle
        ctx.beginPath();
        ctx.fillStyle = `rgba(255, 0, 0, ${0.3 * (1 - progress)})`;
        ctx.arc(effect.x, effect.y, 40 * (1 - progress), 0, Math.PI * 2);
        ctx.fill();
        
        // Particle explosion
        for (let i = 0; i < 12; i++) {
            const angle = (i / 12) * Math.PI * 2 + progress * Math.PI;
            const distance = progress * 50;  // Larger distance
            const x = effect.x + Math.cos(angle) * distance;
            const y = effect.y + Math.sin(angle) * distance;
            
            // Larger particles
            ctx.beginPath();
            ctx.fillStyle = `rgba(255, ${100 - progress * 100}, ${100 - progress * 100}, ${1 - progress})`;
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
            
            // Trailing effect
            ctx.beginPath();
            ctx.strokeStyle = `rgba(255, 0, 0, ${(1 - progress) * 0.5})`;
            ctx.lineWidth = 2;
            ctx.moveTo(effect.x, effect.y);
            ctx.lineTo(x, y);
            ctx.stroke();
        }
        
        // "DIED" text
        ctx.font = 'bold 20px Arial';
        ctx.fillStyle = `rgba(255, 0, 0, ${1 - progress})`;
        ctx.textAlign = 'center';
        ctx.fillText('💀', effect.x, effect.y - 40 * progress);
        
        ctx.restore();
    }
}

// Update the click handler to use current ws instance
canvas.onclick = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const rect = canvas.getBoundingClientRect();
        ws.send(JSON.stringify({
            type: 'add_food',
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        }));
    }
};

function drawBlobFish(blob) {
    ctx.save();
    ctx.translate(blob.x, blob.y);
    
    // Shadow for depth
    ctx.shadowColor = 'rgba(0,0,0,0.2)';
    ctx.shadowBlur = 15;
    
    // Body gradient
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 40);
    gradient.addColorStop(0, '#FFD6D6');  // Lighter pink center
    gradient.addColorStop(0.6, '#FAB0B0');  // Original pink
    gradient.addColorStop(1, '#F29B9B');  // Darker edge
    
    // Main blob body
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(-40, 0);
    ctx.bezierCurveTo(-40, -30, 40, -30, 40, 0);
    ctx.bezierCurveTo(40, 30, -40, 30, -40, 0);
    ctx.fill();
    
    // Nerdy glasses with glare
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 2.5;
    
    // Left lens
    ctx.beginPath();
    ctx.ellipse(-12, -5, 10, 12, -0.1, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.fill();
    
    // Right lens
    ctx.beginPath();
    ctx.ellipse(12, -5, 10, 12, 0.1, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fill();
    
    // Glasses bridge
    ctx.beginPath();
    ctx.moveTo(-2, -5);
    ctx.lineTo(2, -5);
    ctx.stroke();
    
    // Glasses arms
    ctx.beginPath();
    ctx.moveTo(-22, -5);
    ctx.lineTo(-35, -8);
    ctx.moveTo(22, -5);
    ctx.lineTo(35, -8);
    ctx.stroke();
    
    // Glare on glasses
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(-15, -8, 3, 0, Math.PI * 0.8);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(9, -8, 3, 0, Math.PI * 0.8);
    ctx.stroke();
    
    // Cute academic expression
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    
    // Slightly concerned eyebrows
    ctx.beginPath();
    ctx.moveTo(-18, -22);
    ctx.quadraticCurveTo(-15, -24, -8, -22);
    ctx.moveTo(18, -22);
    ctx.quadraticCurveTo(15, -24, 8, -22);
    ctx.stroke();
    
    // Small thoughtful mouth
    ctx.beginPath();
    ctx.arc(0, 10, 8, 0.2, Math.PI - 0.2);
    ctx.stroke();
    
    // Tail with more character
    ctx.save();
    ctx.translate(-40, 0);
    ctx.rotate(blob.tail_angle);
    const tailGradient = ctx.createLinearGradient(-20, 0, 0, 0);
    tailGradient.addColorStop(0, '#F29B9B');
    tailGradient.addColorStop(1, '#FAB0B0');
    ctx.fillStyle = tailGradient;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(-15, -15, -20, 0);
    ctx.quadraticCurveTo(-15, 15, 0, 0);
    ctx.fill();
    ctx.restore();
    
    // Optional: Add a tiny graduation cap
    ctx.fillStyle = '#333';
    ctx.beginPath();
    ctx.moveTo(-15, -35);
    ctx.lineTo(15, -35);
    ctx.lineTo(15, -32);
    ctx.lineTo(-15, -32);
    ctx.fill();
    ctx.fillRect(-10, -40, 20, 5);
    
    // Tassel
    ctx.strokeStyle = '#FFD700';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(10, -35);
    ctx.quadraticCurveTo(15, -30, 18, -25);
    ctx.stroke();
    ctx.fillStyle = '#FFD700';
    ctx.beginPath();
    ctx.arc(18, -25, 2, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.restore();
}

// Add dev thoughts panel to HTML
const devPanel = document.createElement('div');
devPanel.className = 'dev-panel';
devPanel.innerHTML = `
    <div class="dev-title">🤓 Dev Blob's Observations</div>
    <div id="devThoughts"></div>
`;
document.querySelector('.data-panel').prepend(devPanel);

function updateDevPanel(thought) {
    const devThoughts = document.getElementById('devThoughts');
    if (!thought || !devThoughts) return;
    
    // Create a unique key for the thought by combining timestamp and content
    const thoughtKey = `${thought.slice(0, 50)}`;  // Use first 50 chars as key
    
    // Don't add if this exact thought exists
    if (Array.from(devThoughts.children).some(elem => 
        elem.getAttribute('data-thought') === thoughtKey)) {
        return;
    }
    
    // Clear panel if it's a report (longer thought)
    if (thought.length > 50) {
        devThoughts.innerHTML = '';
    }
    
    const thoughtElem = document.createElement('div');
    thoughtElem.className = 'dev-thought';
    thoughtElem.setAttribute('data-thought', thoughtKey);
    
    // Add timestamp for regular observations
    if (thought.length <= 50) {
        const time = new Date().toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit'
        });
        thoughtElem.textContent = `[${time}] ${thought}`;
    } else {
        thoughtElem.textContent = thought;
    }
    
    devThoughts.insertBefore(thoughtElem, devThoughts.firstChild);
    
    // Keep only last 3 observations (not reports)
    const observations = Array.from(devThoughts.children).filter(
        elem => elem.textContent.includes('[')
    );
    
    if (observations.length > 3) {
        observations[3].remove();
    }
}

function drawDevThought(ctx, blob) {
    if (!blob.thought) return;
    
    ctx.save();
    
    // Position above blob fish
    const bubbleX = blob.x;
    const bubbleY = blob.y - 60;
    
    // Different styling for reports vs quick thoughts
    const isReport = blob.thought.length > 50;
    
    // Setup text style
    ctx.font = isReport ? '10px Arial' : 'bold 14px Arial';
    
    // Word wrap for reports
    const maxWidth = 200;  // Maximum width for text
    const lineHeight = isReport ? 12 : 20;
    const words = blob.thought.split(' ');
    let lines = [];
    let currentLine = words[0];
    
    // Word wrap
    for (let i = 1; i < words.length; i++) {
        const testLine = currentLine + ' ' + words[i];
        const metrics = ctx.measureText(testLine);
        if (metrics.width > maxWidth) {
            lines.push(currentLine);
            currentLine = words[i];
        } else {
            currentLine = testLine;
        }
    }
    lines.push(currentLine);
    
    // Calculate bubble dimensions
    const bubbleWidth = maxWidth + 20;
    const bubbleHeight = lines.length * lineHeight + 20;
    
    // Draw bubble with gradient
    const gradient = ctx.createLinearGradient(
        bubbleX - bubbleWidth/2, bubbleY,
        bubbleX + bubbleWidth/2, bubbleY
    );
    gradient.addColorStop(0, 'rgba(255, 240, 240, 0.95)');
    gradient.addColorStop(1, 'rgba(250, 220, 220, 0.95)');
    
    ctx.fillStyle = gradient;
    ctx.strokeStyle = 'rgba(200, 100, 100, 0.5)';
    ctx.lineWidth = 2;
    
    // Bubble
    ctx.beginPath();
    ctx.roundRect(
        bubbleX - bubbleWidth/2,
        bubbleY - bubbleHeight/2,
        bubbleWidth,
        bubbleHeight,
        8
    );
    ctx.fill();
    ctx.stroke();
    
    // Tail
    ctx.beginPath();
    ctx.moveTo(bubbleX - 10, bubbleY + bubbleHeight/2);
    ctx.lineTo(bubbleX, bubbleY + bubbleHeight/2 + 10);
    ctx.lineTo(bubbleX + 10, bubbleY + bubbleHeight/2);
    ctx.fillStyle = 'rgba(250, 220, 220, 0.95)';
    ctx.fill();
    ctx.stroke();
    
    // Text
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Draw each line
    lines.forEach((line, i) => {
        const y = bubbleY - (lines.length * lineHeight)/2 + (i + 0.5) * lineHeight;
        ctx.fillText(line, bubbleX, y);
    });
    
    ctx.restore();
}

// Add viewer count display
function updateViewerCount(count) {
    const viewerDisplay = document.getElementById('viewerCount') || createViewerDisplay();
    if (count >= 5) {
        viewerDisplay.textContent = `👥 ${count} watching`;
        viewerDisplay.style.display = 'block';
    } else {
        viewerDisplay.style.display = 'none';
    }
}

function createViewerDisplay() {
    const display = document.createElement('div');
    display.id = 'viewerCount';
    display.className = 'viewer-count';
    document.querySelector('.container').prepend(display);
    return display;
}