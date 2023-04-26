// Flight Simulator
// Authors: Bryan Cohen & Tanishq Iyer
'use strict';

import { calcNormals } from "./tools.js";
import { QuadTree, AxisAlignedBoundingBox } from "./spacePartitioning.js";

const vec2 = glMatrix.vec2;
const vec3 = glMatrix.vec3;
// const vec4 = glMatrix.vec4;
const mat4 = glMatrix.mat4;
const quat = glMatrix.quat;

// Global WebGL context variable
let gl;

window.addEventListener('load', function init() {
    // Get the HTML5 canvas object from it's ID
    const canvas = document.getElementById('webgl-canvas');
    if (!canvas) { window.alert('Could not find #webgl-canvas'); return; }

    // Get the WebGL context (save into a global variable)
    gl = canvas.getContext('webgl2');
    if (!gl) { window.alert("WebGL isn't available"); return; }

    // Configure WebGL
    gl.clearColor(0.1, 0.8, 1.0, 1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    
    // Initialize the WebGL program and data
    gl.program = initProgram();
    initBuffers();

    initEvents();
    onWindowResize();

    // Start the Game Loop
    runFrame();
});

/**
 * Initializes the WebGL program.
 */
function initProgram() {
    // Compile shaders
    // Vertex Shader
    let vert_shader = compileShader(gl, gl.VERTEX_SHADER,
        `#version 300 es
        precision mediump float;

        uniform vec4 uLight; // Light position
        uniform float uLightIntensity;

        uniform mat4 uModelMatrix;
        uniform mat4 uCameraMatrix;
        uniform mat4 uProjectionMatrix;

        in vec4 aPosition;
        in vec3 aNormal;
        in vec3 aColor;

        out vec3 vNormalVector;
        out vec3 vLightVector;
        out vec3 vEyeVector;
        out float vLightIntensity;

        flat out vec3 vColor;

        void main() {
            mat4 viewMatrix = inverse(uCameraMatrix);
            mat4 modelViewMatrix = viewMatrix * uModelMatrix;

            vec4 P = modelViewMatrix * aPosition;

            vNormalVector = mat3(modelViewMatrix) * aNormal;
            vec4 light = viewMatrix * uLight;
            vLightVector = (light.w == 1.0) ? (P - light).xyz : light.xyz;
            vEyeVector = -P.xyz; // from position to camera
            vLightIntensity = uLightIntensity;

            gl_Position = uProjectionMatrix * P;
            vColor = aColor;
        }`
    );

    // Fragment Shader
    let frag_shader = compileShader(gl, gl.FRAGMENT_SHADER,
        `#version 300 es
        precision mediump float;

        // Light constants
        const vec3 lightColor = normalize(vec3(1.0, 1.0, 1.0));
        in float vLightIntensity;

        // Material constants
        const float materialAmbient = 0.2;
        const float materialDiffuse = 0.95;
        const float materialSpecular = 0.05;
        const float materialShininess = 5.0;
        
        // Attenuation constants
        const float lightConstantA = 0.05;
        const float lightConstantB = 0.02;
        const float lightConstantC = 0.02;

        in vec3 vNormalVector;
        in vec3 vLightVector;
        in vec3 vEyeVector;

        // Fragment base color
        flat in vec3 vColor;

        // Output color of the fragment
        out vec4 fragColor;

        void main() {
            // normalize vectors
            vec3 N = normalize(vNormalVector);
            vec3 L = normalize(vLightVector);
            vec3 E = normalize(vEyeVector);

            float diffuse = dot(-L, N);
            float specular = 0.0;
            if (diffuse < 0.0) {
                diffuse = 0.0;
            } else {
                vec3 R = reflect(L, N);
                specular = pow(max(dot(R, E), 0.0), materialShininess);
            }

            float d = length(vLightVector);
            float attenuation = 1.0 / ((lightConstantA * d * d) + (lightConstantB * d) + lightConstantC);

            // compute lighting
            float A = materialAmbient;
            float D = materialDiffuse * diffuse * attenuation;
            float S = materialSpecular * specular * attenuation;

            fragColor.rgb = (((A + D) * vColor) + S) * lightColor * vLightIntensity;
            fragColor.a = 1.0;
        }
        `
    );

    // Link the shaders into a program and use them with the WebGL context
    let program = linkProgram(gl, vert_shader, frag_shader);
    gl.useProgram(program);
    
    // Get the attribute indices
    program.aPosition = gl.getAttribLocation(program, 'aPosition');
    program.aNormal = gl.getAttribLocation(program, 'aNormal');
    program.aColor = gl.getAttribLocation(program, 'aColor');

    // Get the uniform indices
    program.uCameraMatrix = gl.getUniformLocation(program, 'uCameraMatrix');
    program.uModelMatrix = gl.getUniformLocation(program, 'uModelMatrix');
    program.uProjectionMatrix = gl.getUniformLocation(program, 'uProjectionMatrix');
    program.uLight = gl.getUniformLocation(program, 'uLight');
    program.uLightIntensity = gl.getUniformLocation(program, 'uLightIntensity');

    return program;
}

/**
 * Initialize the camera and the data buffers.
 * Also initialize the octree.
 */
function initBuffers() {
    gl.world = createSceneTreeNode("world");

    // Create the camera
    const camera = createSceneTreeNode("camera");
    {
        const t = camera.localTransform;
        const translation = mat4.fromTranslation(mat4.create(), [0, 0, -1]);
        
        mat4.multiply(t, t, translation);
    }

    // Load drone model into GPU
    {
        const cube_coords = [
            1, 1, 1, // A
            -1, 1, 1, // B
            -1, -1, 1, // C
            1, -1, 1, // D
            1, -1, -1, // E
            -1, -1, -1, // F
            -1, 1, -1, // G
            1, 1, -1, // H
        ];

        const cube_colors = [
            1, 0, 0, // red
            1, 1, 0, // yellow
            0, 1, 0, // green
            0, 0, 0, // black (color is not actually used)
            0, 1, 1, // cyan
            0, 0, 1, // blue
            0, 0, 0, // black (color is not actually used)
            1, 0, 1, // purple
        ];

        const cube_indices = [
            1, 2, 0, 2, 3, 0,
            7, 6, 1, 0, 7, 1,
            1, 6, 2, 6, 5, 2,
            3, 2, 4, 2, 5, 4,
            6, 7, 5, 7, 4, 5,
            0, 3, 7, 3, 4, 7,
        ];

        gl.drone = createSceneTreeNode("model");
        gl.drone.model = loadModel(cube_coords, cube_colors, cube_indices);
        
        // compute drone's transform
        const t = gl.drone.localTransform;
        scaleByConstant(t, 0.2);
    }

    // Generate and load terrain into GPU
    {
        gl.terrain = createSceneTreeNode("model");

        const seed = 782378;
        const terrain = generate_terrain(7, 0.01, Math.seedrandom(seed));
        let [terrainVertices, terrainIndices] = generate_mesh(terrain);
        gl.terrain.model = loadModel(terrainVertices, null, terrainIndices, true);

        // compute terrain's transform
        const t = gl.terrain.localTransform;
        mat4.multiply(t, t, mat4.fromScaling(mat4.create(), [20, 10, 20]));

        // flip terrain upside-down (inverted by default)
        mat4.multiply(t, t, angleAxisToMat4(180, [0, 0, 1]));
    }

    gl.world.addChild(gl.terrain);
    gl.quadtree = QuadTree(
        gl.terrain.model,
        gl.terrain.transform,
        [0, 0, 0],
        [40, 0, 40],
        8
    );

    gl.world.addChild(gl.drone);
    gl.drone.addChild(camera);

    gl.world.camera = camera;
}

/**
 * Creates a default scene tree node.
 */
function createSceneTreeNode(type) {
    let obj = {
        type: type,
        localTransform: mat4.identity(mat4.create()),
        parent: null,
        children: []
    };

    obj.addChild = function addChild(child) {
        if (child.parent !== null) {
            throw new Error("Child already has a parent.");
        }

        child.parent = this;
        this.children.push(child);
    }

    Object.defineProperty(obj, "transform", {
        get: function () {
            if (this.parent === null) {
                return this.localTransform;
            } else {
                const t = mat4.multiply(
                    mat4.create(),
                    this.parent.transform,
                    this.localTransform
                );
                return t;
            }
        }
    });

    return obj;
}

/**
 * Initialize event handlers
 */
function initEvents() {
    // stores which keys are pressed
    gl.input = {
        state: {},
        reset: function () { this.state = {}; },
        isKeyDown: function (key) {
            return this.state[key] === true;
        }
    };

    window.addEventListener('resize', onWindowResize);

    window.addEventListener('keydown', function (e) {
        e.preventDefault();
        gl.input.state[e.key] = true;
    });

    window.addEventListener('keyup', function (e) {
        e.preventDefault();
        gl.input.state[e.key] = false;
    });
}

/**
 * Keep the canvas sized to the window.
 */
function onWindowResize() {
    const size = Math.min(window.innerWidth, window.innerHeight);

    gl.canvas.width = size;
    gl.canvas.height = size;
    gl.viewport(0, 0, size, size);

    updateProjectionMatrix();
}

/**
 * Updates the projection transformation matrix.
 */
function updateProjectionMatrix() {
    let [w, h] = [gl.canvas.width, gl.canvas.height];

    const mv = mat4.perspective(mat4.create(), degreesToRadians(90), w / h, 0.0001, 1000);

    // for debugging
    // const mv = mat4.ortho(mat4.create(), -2, 2, -2, 2, 1000, -1000);

    gl.uniformMatrix4fv(gl.program.uProjectionMatrix, false, mv);
}

/**
 * Runs all tasks for a single frame.
 */
function runFrame() {
    console.log("FRAME START");

    updateDroneTransform();

    render();

    window.requestAnimationFrame(runFrame);
}

/**
 * Render the scene.
 */
function render() {
    const ms = performance.now();

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Calculate directional light for day/night cycle
    {
        const dayNightPeriodInSecs = 300;
        const dayNightRatio = ((ms / 1000) % dayNightPeriodInSecs) / dayNightPeriodInSecs;
        let angle = dayNightRatio * 360;

        const baseIntensity = 0.5;
        const extraDayIntensity = 2;
        let intensity;
        
        if (angle < 180) {
            intensity = baseIntensity + (extraDayIntensity * Math.sin(degreesToRadians(angle)));
        } else {
            angle = 360 - angle;
            intensity = baseIntensity;
        }
        
        const dir = [0, 0, -1];
        vec3.transformMat4(dir, dir, angleAxisToMat4(angle, [1, 0, 0]));
        
        gl.uniform4fv(gl.program.uLight, [...dir, 0]);
        gl.uniform1f(gl.program.uLightIntensity, intensity);
    }
    
    gl.uniformMatrix4fv(gl.program.uCameraMatrix, false, gl.world.camera.transform);

    const draw = function (obj) {
        if (obj.type === "model") {
            gl.uniformMatrix4fv(gl.program.uModelMatrix, false, obj.transform);

            const model = obj.model;
            gl.bindVertexArray(model.vao);
            gl.drawElements(model.mode, model.count, gl.UNSIGNED_SHORT, 0);
        }

        for (const child of obj.children) {
            draw(child);
        }
    }

    draw(gl.world);

    // Cleanup
    gl.bindVertexArray(null);
}

function updateDroneTransform() {
    // updated is in local space
    const delta = mat4.identity(mat4.create());

    // !== for booleans does exclusive or

    // moves forward and backward
    if (gl.input.isKeyDown("ArrowUp") !== gl.input.isKeyDown("ArrowDown")) {
        const factor = gl.input.isKeyDown("ArrowUp") ? -1 : 1;
        const t = mat4.fromTranslation(mat4.create(), [0, 0, 0.2 * factor]);
        mat4.multiply(delta, delta, t);
    }

    const rotations = [];

    // yaw (rotate y-axis)
    if (gl.input.isKeyDown("ArrowLeft") !== gl.input.isKeyDown("ArrowRight")) {
        const factor = gl.input.isKeyDown("ArrowRight") ? 1 : -1;
        rotations.push(angleAxisToQuat(1 * factor, [0, 1, 0]));
    }

    // pitch (rotate x-axis)
    if (gl.input.isKeyDown("w") !== gl.input.isKeyDown("s")) {
        const factor = gl.input.isKeyDown("s") ? 1 : -1;
        rotations.push(angleAxisToQuat(1 * factor, [1, 0, 0]));
    }

    // roll (rotate z-axis)
    if (gl.input.isKeyDown("a") !== gl.input.isKeyDown("d")) {
        const factor = gl.input.isKeyDown("d") ? 1 : -1;
        rotations.push(angleAxisToQuat(1 * factor, [0, 0, 1]));
    }

    let finalRotation = rotations.reduce((a, b) => quat.multiply(a, a, b),
        quat.identity(quat.create()));
    finalRotation = mat4.fromQuat(mat4.create(), finalRotation);

    mat4.multiply(delta, delta, finalRotation);

    const updatedInWorldSpace = mat4.create();
    mat4.multiply(updatedInWorldSpace, gl.drone.transform, delta);
    {
        const s = mat4.fromScaling(mat4.create(), Array(3).fill().map(() => 1.1));
        mat4.multiply(updatedInWorldSpace, s, updatedInWorldSpace);
    }
    const collidesWithTerrain = gl.quadtree.checkCollision(gl.drone.model, updatedInWorldSpace);

    const terrainAABB = AxisAlignedBoundingBox(gl.terrain.model, gl.terrain.transform);
    const droneAABB = AxisAlignedBoundingBox(gl.drone.model, updatedInWorldSpace);

    const outOfBounds = droneAABB.max[0] > terrainAABB.max[0] ||
                        droneAABB.max[2] > terrainAABB.max[2] ||
                        droneAABB.min[0] < terrainAABB.min[0] ||
                        droneAABB.min[2] < terrainAABB.min[2];

    if (!(collidesWithTerrain || outOfBounds)) {
        // if no collisions, then move the drone
        const t = gl.drone.localTransform;
        mat4.multiply(t, t, delta);
    }
}

/**
 * Loads a model into GPU with the coordinates, colors, and indices provided.
 */
function loadModel(coords, colors, indices, useStrips) {
    useStrips = useStrips === true;

    // Create and bind VAO
    let vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    
    // Load vertex positions into GPU
    coords = Float32Array.from(coords);
    loadArrayBuffer(coords, gl.program.aPosition, 3, gl.FLOAT);

    // Load vertex normals into GPU
    const normals = calcNormals(coords, indices, useStrips)
    loadArrayBuffer(normals, gl.program.aNormal, 3, gl.FLOAT);

    if (colors === null) {
        colors = Array(coords.length / 3).fill().flatMap(() => [0, 1, 0]);
    }
    
    // Load vertex colors into GPU
    loadArrayBuffer(Float32Array.from(colors), gl.program.aColor, 3, gl.FLOAT);

    // Load the index data into the GPU
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint16Array.from(indices), gl.STATIC_DRAW);

    // Cleanup
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    const count = indices.length;
    const mode = useStrips ? gl.TRIANGLE_STRIP : gl.TRIANGLES;

    const object = { vao, count, mode, coords };

    return object;
}

/**
 * Creates and loads an array buffer into the GPU.
 * Then attaches it to a location and enables it.
 * values - an array of values to upload to the buffer.
 * location - the location the buffer should attach to.
 * numComponents - the number of components per attribute.
 * numType - the type of the component.
 */
function loadArrayBuffer(values, location, numComponents, componentType) {
    const buf = gl.createBuffer();
    
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, values, gl.STATIC_DRAW);
    gl.vertexAttribPointer(location, numComponents, componentType, false, 0, 0);
    gl.enableVertexAttribArray(location);

    return buf;
}

/**
 * converts from angle-axis to quaternion.
 * angle: angle in degrees.
 * axis: a vec3 representing a direction.
 */
function angleAxisToQuat(angle, axis) {
    angle = degreesToRadians(angle);
    
    axis = vec3.normalize(vec3.create(), axis);

    const sin = Math.sin(angle/2);
    const cos = Math.cos(angle/2);

    const q = quat.fromValues(
        axis[0] * sin,
        axis[1] * sin,
        axis[2] * sin,
        cos
    );

    return q;
}

function angleAxisToMat4(angle, axis) {
    const q = angleAxisToQuat(angle, axis);
    
    return mat4.fromQuat(mat4.create(), q);
}

// Converts an angle in degrees to radians.
function degreesToRadians(angle) {
    return (angle / 360) * (2 * Math.PI);
}

// scales a mat4 by a constant.
function scaleByConstant(matrix, value) {
    const scaling = mat4.fromScaling(
        mat4.create(),
        Array(3).fill(value)
    );

    mat4.multiply(matrix, matrix, scaling);
}
