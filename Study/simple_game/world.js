function createGameWorld(container) {
    const world = {
        obstacles: [],
        buildings: [],
        ponds: [],
        captureObjects: [],
        npcs: []
    };

    const terrainTypes = ['Grass', 'Pond'];
    const gridSize = 50; // Size of each tile

    // Create terrain as a grid
    for (let x = 0; x < container.clientWidth; x += gridSize) {
        for (let y = 0; y < container.clientHeight; y += gridSize) {
            const terrainType = 'Grass';
            const terrain = new Entity(container, 'terrain', x, y, terrainType);
            terrain.element.style.width = `${gridSize}px`;
            terrain.element.style.height = `${gridSize}px`;
            terrain.element.innerText = terrainType;
            // Grass is not added to obstacles
        }
    }

    // Create ponds as grid-aligned ovals
    for (let i = 0; i < 3; i++) {
        const pondWidth = 3; // 3 tiles wide
        const pondHeight = 2; // 2 tiles high
        const pondX = Math.floor(Math.random() * (container.clientWidth / gridSize - pondWidth)) * gridSize;
        const pondY = Math.floor(Math.random() * (container.clientHeight / gridSize - pondHeight)) * gridSize;
        for (let x = 0; x < pondWidth; x++) {
            for (let y = 0; y < pondHeight; y++) {
                const pond = new Entity(container, 'pond', pondX + x * gridSize, pondY + y * gridSize, 'Pond');
                pond.element.style.width = `${gridSize}px`;
                pond.element.style.height = `${gridSize}px`;
                pond.element.style.borderRadius = '50%';
                world.ponds.push(pond);
                world.obstacles.push(pond); // Ponds are added to obstacles
            }
        }
    }

    // Create some buildings (N by N)
    const buildingSize = 2; // N by N
    for (let i = 0; i < 5; i++) {
        const buildingX = Math.floor(Math.random() * (container.clientWidth / gridSize - buildingSize)) * gridSize;
        const buildingY = Math.floor(Math.random() * (container.clientHeight / gridSize - buildingSize)) * gridSize;
        for (let x = 0; x < buildingSize; x++) {
            for (let y = 0; y < buildingSize; y++) {
                const building = new Entity(container, 'building', buildingX + x * gridSize, buildingY + y * gridSize, 'Building');
                building.element.style.width = `${gridSize}px`;
                building.element.style.height = `${gridSize}px`;
                world.buildings.push(building);
                world.obstacles.push(building); // Buildings are added to obstacles
            }
        }
    }

    // Create game world boundaries
    const boundaryThickness = 10; // Thickness of the boundary

    // Top boundary
    const topBoundary = new Entity(container, 'boundary', 0, -boundaryThickness, 'Boundary');
    topBoundary.element.style.width = `${container.clientWidth}px`;
    topBoundary.element.style.height = `${boundaryThickness}px`;
    world.obstacles.push(topBoundary);

    // Bottom boundary
    const bottomBoundary = new Entity(container, 'boundary', 0, container.clientHeight, 'Boundary');
    bottomBoundary.element.style.width = `${container.clientWidth}px`;
    bottomBoundary.element.style.height = `${boundaryThickness}px`;
    world.obstacles.push(bottomBoundary);

    // Left boundary
    const leftBoundary = new Entity(container, 'boundary', -boundaryThickness, 0, 'Boundary');
    leftBoundary.element.style.width = `${boundaryThickness}px`;
    leftBoundary.element.style.height = `${container.clientHeight}px`;
    world.obstacles.push(leftBoundary);

    // Right boundary
    const rightBoundary = new Entity(container, 'boundary', container.clientWidth, 0, 'Boundary');
    rightBoundary.element.style.width = `${boundaryThickness}px`;
    rightBoundary.element.style.height = `${container.clientHeight}px`;
    world.obstacles.push(rightBoundary);

    // Create capture objects at specific positions
    const captureObjectPositions = [
        { x: container.clientWidth / 4, y: container.clientHeight / 4 }, // Top-left
        { x: 3 * container.clientWidth / 4, y: container.clientHeight / 4 }, // Top-right
        { x: container.clientWidth / 4, y: 3 * container.clientHeight / 4 }, // Bottom-left
        { x: 3 * container.clientWidth / 4, y: 3 * container.clientHeight / 4 } // Bottom-right
    ];

    captureObjectPositions.forEach(pos => {
        const captureX = Math.floor(pos.x / gridSize) * gridSize;
        const captureY = Math.floor(pos.y / gridSize) * gridSize;
        const captureObject = new CaptureObject(container, captureX, captureY);
        world.captureObjects.push(captureObject);
        world.obstacles.push(captureObject);
    });

    // Create NPCs
    const teams = gameConfig.maxTeams;
    const npcsPerTeam = Math.floor(gameConfig.maxNPCs / teams);
    for (let team = 1; team <= teams; team++) {
        for (let i = 0; i < npcsPerTeam; i++) {
            const npcX = Math.random() * container.clientWidth;
            const npcY = Math.random() * container.clientHeight;
            const npc = new NPC(container, npcX, npcY, team, behaviorTree);
            world.npcs.push(npc);
        }
    }

    return world;
}
