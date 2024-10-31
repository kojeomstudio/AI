console.log('entities.js loaded');

let entityIdCounter = 0; // Global counter for unique IDs

class Entity {
    constructor(container, className, x, y, name) {
        this.id = ++entityIdCounter; // Assign a unique ID
        this.container = container;
        this.element = document.createElement('div');
        this.element.className = `entity ${className}`;
        this.element.style.left = `${x}px`;
        this.element.style.top = `${y}px`;

        const label = document.createElement('div');
        label.innerText = name;
        this.element.appendChild(label);

        container.appendChild(this.element);
        this.x = x;
        this.y = y;
        this.width = this.element.offsetWidth;
        this.height = this.element.offsetHeight;
        this.containerWidth = container.clientWidth;
        this.containerHeight = container.clientHeight;
    }

    move(dx, dy) {
        const newX = this.x + dx;
        const newY = this.y + dy;

        if (newX >= 0 && newX <= this.containerWidth - this.width) {
            this.x = newX;
        }
        if (newY >= 0 && newY <= this.containerHeight - this.height) {
            this.y = newY;
        }

        this.element.style.left = `${this.x}px`;
        this.element.style.top = `${this.y}px`;
    }

    checkCollision(otherEntity) {
        return !(
            this.x > otherEntity.x + otherEntity.width ||
            this.x + this.width < otherEntity.x ||
            this.y > otherEntity.y + otherEntity.height ||
            this.y + this.height < otherEntity.y
        );
    }
}

class Character extends Entity {
    constructor(container, className, x, y, name, behaviorTree) {
        super(container, className, x, y, name);
        this.behaviorTree = behaviorTree;

        this.actionLabel = document.createElement('div');
        this.actionLabel.className = 'action-label';
        this.element.appendChild(this.actionLabel);
        this.currentAction = 'wander';
        this.updateActionLabel();

        this.wanderDirection = { dx: 0, dy: 0 };
        this.wanderDuration = 0;
        this.lifespan = gameConfig.despawnLifespan.min + Math.random() * (gameConfig.despawnLifespan.max - gameConfig.despawnLifespan.min);
    }

    act(player, obstacles, entities) {
        const distX = Math.abs(player.x - this.x);
        const distY = Math.abs(player.y - this.y);
        const distance = Math.sqrt(distX * distX + distY * distY);

        let action = this.behaviorTree.default.action;
        for (const transition of this.behaviorTree.default.transitions) {
            if (transition.condition === "player_nearby" && distance < 100) {
                action = transition.action;
            } else if (transition.condition === "player_approaching" && distance < 200) {
                action = transition.action;
            } else if (transition.condition === "obstacle_nearby" && this.isObstacleNearby(obstacles)) {
                action = transition.action;
            } else if (transition.condition === "out_of_bounds" && this.isOutOfBounds()) {
                action = transition.action;
            }
        }

        this.currentAction = action;
        this.updateActionLabel();

        switch (action) {
            case "wander":
                this.moveRandomly();
                break;
            case "flee":
                this.move(-distX / distance * gameConfig.monsterSpeed, -distY / distance * gameConfig.monsterSpeed);
                break;
            case "keep_distance":
                if (distance < 100) {
                    this.move(-distX / distance * gameConfig.monsterSpeed, -distY / distance * gameConfig.monsterSpeed);
                } else {
                    this.moveRandomly();
                }
                break;
            case "avoid_obstacle":
                this.avoidObstacle(obstacles);
                break;
            case "stay_within_bounds":
                this.moveTowardsCenter();
                break;
        }

        if (this.checkCollision(player)) {
            console.log('Collision with player');
        }

        entities.forEach(otherEntity => {
            if (this !== otherEntity && this.checkCollision(otherEntity)) {
                console.log('Collision with another entity');
            }
        });

        obstacles.forEach(obstacle => {
            if (this.checkCollision(obstacle)) {
                console.log('Collision with an obstacle');
            }
        });

        this.lifespan -= 1;
        if (this.lifespan <= 0) {
            this.despawn();
        }
    }

    moveRandomly() {
        if (this.wanderDuration <= 0) {
            const angle = Math.random() * 2 * Math.PI;
            this.wanderDirection = {
                dx: Math.cos(angle) * gameConfig.monsterSpeed,
                dy: Math.sin(angle) * gameConfig.monsterSpeed
            };
            this.wanderDuration = 50 + Math.random() * 100;
        }

        this.move(this.wanderDirection.dx, this.wanderDirection.dy);
        this.wanderDuration -= 1;
    }

    updateActionLabel() {
        this.actionLabel.innerText = this.currentAction;
    }

    despawn() {
        if (this.element.parentNode === this.container) {
            this.container.removeChild(this.element);
            console.log(`${this.constructor.name} ${this.id} despawned`);
        } else {
            console.error(`${this.constructor.name} not found in container`);
        }
    }

    isObstacleNearby(obstacles) {
        return obstacles.some(obstacle => this.checkCollision(obstacle));
    }

    avoidObstacle(obstacles) {
        obstacles.forEach(obstacle => {
            if (this.checkCollision(obstacle)) {
                const dx = this.x - obstacle.x;
                const dy = this.y - obstacle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                this.move(dx / distance * gameConfig.monsterSpeed, dy / distance * gameConfig.monsterSpeed);
                return;
            }
        });
    }

    isOutOfBounds() {
        return this.x < 0 || this.x > this.containerWidth || this.y < 0 || this.y > this.containerHeight;
    }

    moveTowardsCenter() {
        const centerX = this.containerWidth / 2;
        const centerY = this.containerHeight / 2;
        const dx = centerX - this.x;
        const dy = centerY - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        this.move(dx / distance * gameConfig.monsterSpeed, dy / distance * gameConfig.monsterSpeed);
    }
}

class NPC extends Character {
    constructor(container, x, y, team, behaviorTree) {
        super(container, `npc team-${team}`, x, y, `NPC ${entityIdCounter} (Team ${team})`, behaviorTree);
        this.team = team;
        this.capturingObject = null;
        this.captureProgress = 0;
    }

    // Additional methods specific to NPCs
}

class Monster extends Character {
    constructor(container, x, y, behaviorTree) {
        super(container, 'monster', x, y, `Monster ${entityIdCounter}`, behaviorTree);
        this.health = 10;
    }

    // Additional methods specific to Monsters
}

class Player extends Entity {
    constructor(container, x, y) {
        super(container, 'player', x, y, 'Player');
    }

    attack(monster) {
        console.log('Player attacks monster');
    }
}

class CaptureObject extends Entity {
    constructor(container, x, y) {
        super(container, 'capture-object', x, y, `Capture ${entityIdCounter}`);
        this.owner = null;
        this.capturingNPC = null;
        this.captureProgress = 0;

        this.statusLabel = document.createElement('div');
        this.statusLabel.className = 'action-label';
        this.element.insertBefore(this.statusLabel, this.element.firstChild); // Insert before the name label
        this.updateLabel();
    }

    capture(npc) {
        if (this.capturingNPC === npc) {
            this.captureProgress++;
            if (this.captureProgress >= gameConfig.captureTime) {
                this.owner = npc;
                this.capturingNPC = null;
                this.captureProgress = 0;
                this.updateLabel();
            }
        } else {
            this.capturingNPC = npc;
            this.captureProgress = 0;
        }
    }

    resetCapture() {
        this.capturingNPC = null;
        this.captureProgress = 0;
        this.updateLabel();
    }

    updateLabel() {
        if (this.owner) {
            this.statusLabel.innerText = `Captured by ${this.owner.id}`;
        } else if (this.capturingNPC) {
            this.statusLabel.innerText = 'Capturing...';
        } else {
            this.statusLabel.innerText = 'Uncaptured';
        }
        this.element.querySelector('div:last-child').innerText = `Capture ${this.id}`;
    }
}
