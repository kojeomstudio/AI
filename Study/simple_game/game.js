document.addEventListener("DOMContentLoaded", async () => {
    console.log('game.js loaded');
    const loadingScreen = document.getElementById('loading-screen');
    const loginScreen = document.getElementById('login-screen');
    const gameScreen = document.getElementById('game-screen');
    const loginButton = document.getElementById('login-button');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const gameContainer = document.getElementById('game-container');
    const fpsCounter = document.getElementById('fps-counter');

    let player;
    let monsters = [];
    let npcs = [];
    let obstacles = [];
    let buildings = [];
    let captureObjects = [];
    let lastFrameTime = performance.now();
    let frameTimes = [];

    try {
        await initDB();
        console.log('Database initialized');
        loadingScreen.style.display = 'none';
        loginScreen.style.display = 'flex';
    } catch (error) {
        console.error('Error initializing database:', error);
    }

    loginButton.addEventListener('click', async () => {
        const username = usernameInput.value;
        const password = passwordInput.value;
        if (username && password) {
            try {
                const valid = await checkCredentials(username, password);
                if (valid) {
                    startGame();
                } else {
                    alert('Incorrect password');
                }
            } catch (error) {
                console.error('Error during login:', error);
            }
        } else {
            alert('Please enter both username and password');
        }
    });

    function startGame() {
        try {
            console.log('Starting game');
            loginScreen.style.display = 'none';
            gameScreen.style.display = 'flex';
            const world = createGameWorld(gameContainer);
            obstacles = world.obstacles;
            buildings = world.buildings;
            captureObjects = world.captureObjects;
            npcs = world.npcs;

            player = new Player(gameContainer, 100, 100);
            for (let i = 0; i < 5; i++) {
                spawnMonster();
            }

            for (let i = 0; i < gameConfig.maxNPCs; i++) {
                spawnNPC();
            }

            document.addEventListener('keydown', handleKeydown);
            document.addEventListener('keyup', handleKeyup);
            gameLoop();
        } catch (error) {
            console.error('Error starting game:', error);
        }
    }

    function handleKeydown(event) {
        switch (event.key) {
            case 'ArrowUp':
            case 'w':
            case 'W':
                player.move(0, -10);
                event.preventDefault();
                break;
            case 'ArrowDown':
            case 's':
            case 'S':
                player.move(0, 10);
                event.preventDefault();
                break;
            case 'ArrowLeft':
            case 'a':
            case 'A':
                player.move(-10, 0);
                event.preventDefault();
                break;
            case 'ArrowRight':
            case 'd':
            case 'D':
                player.move(10, 0);
                event.preventDefault();
                break;
            case ' ':
                monsters.forEach(monster => {
                    const distX = Math.abs(player.x - monster.x);
                    const distY = Math.abs(player.y - monster.y);
                    if (distX < 50 && distY < 50) {
                        player.attack(monster);
                    }
                });
                event.preventDefault();
                break;
        }
    }

    function handleKeyup(event) {
        switch (event.key) {
            case 'ArrowUp':
            case 'ArrowDown':
            case 'ArrowLeft':
            case 'ArrowRight':
            case 'w':
            case 'W':
            case 's':
            case 'S':
            case 'a':
            case 'A':
            case 'd':
            case 'D':
                event.preventDefault();
                break;
        }
    }

    function calculateFPS() {
        const now = performance.now();
        const delta = now - lastFrameTime;
        lastFrameTime = now;
        frameTimes.push(delta);

        if (frameTimes.length > 100) {
            frameTimes.shift(); // Keep the last 100 frame times
        }

        const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
        const fps = 1000 / avgFrameTime;
        return fps.toFixed(1);
    }

    function gameLoop() {
        monsters = monsters.filter(monster => {
            if (monster.lifespan > 0) {
                monster.act(player, obstacles.concat(buildings), monsters);
                return true;
            } else {
                monster.despawn();
                return false;
            }
        });

        npcs = npcs.filter(npc => {
            if (npc.lifespan > 0) {
                npc.act(player, obstacles.concat(buildings).concat(captureObjects), npcs);
                return true;
            } else {
                npc.despawn();
                return false;
            }
        });

        captureObjects.forEach(captureObject => {
            captureObject.updateLabel();
        });

        if (monsters.length < gameConfig.maxMonsters) {
            spawnMonster();
        }

        if (npcs.length < gameConfig.maxNPCs) {
            spawnNPC();
        }

        // Camera follow player
        gameContainer.scrollLeft = player.x - gameContainer.clientWidth / 2 + player.width / 2;
        gameContainer.scrollTop = player.y - gameContainer.clientHeight / 2 + player.height / 2;

        const fps = calculateFPS();
        fpsCounter.innerText = `FPS: ${fps}`;

        requestAnimationFrame(gameLoop);
    }

    function spawnMonster() {
        let monsterX, monsterY, validPosition;
        do {
            validPosition = true;
            monsterX = Math.random() * gameContainer.clientWidth;
            monsterY = Math.random() * gameContainer.clientHeight;

            if (monsterX < 0 || monsterX > gameContainer.clientWidth || monsterY < 0 || monsterY > gameContainer.clientHeight) {
                validPosition = false;
            }

            for (const obstacle of obstacles) {
                const distX = Math.abs(monsterX - obstacle.x);
                const distY = Math.abs(monsterY - obstacle.y);
                if (distX < 50 && distY < 50) {
                    validPosition = false;
                    break;
                }
            }
        } while (!validPosition);

        const monster = new Monster(gameContainer, monsterX, monsterY, behaviorTree);
        monsters.push(monster);
    }

    function spawnNPC() {
        let npcX, npcY, validPosition;
        do {
            validPosition = true;
            npcX = Math.random() * gameContainer.clientWidth;
            npcY = Math.random() * gameContainer.clientHeight;

            if (npcX < 0 || npcX > gameContainer.clientWidth || npcY < 0 || npcY > gameContainer.clientHeight) {
                validPosition = false;
            }

            for (const obstacle of obstacles) {
                const distX = Math.abs(npcX - obstacle.x);
                const distY = Math.abs(npcY - obstacle.y);
                if (distX < 50 && distY < 50) {
                    validPosition = false;
                    break;
                }
            }
        } while (!validPosition);

        const team = Math.floor(Math.random() * gameConfig.maxTeams) + 1;
        const npc = new NPC(gameContainer, npcX, npcY, team, behaviorTree);
        npcs.push(npc);
    }
});
