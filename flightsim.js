// Two shapes orbiting
'use strict';

// Allow use of glMatrix values directly instead of needing the glMatrix prefix
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
    // gl.enable(gl.CULL_FACE);
    
    // Initialize the WebGL program and data
    gl.program = initProgram();
    initBuffers();

    gl.inputState = {
        state: {},
        reset: () => { gl.inputState.state = {}; },
        isKeyDown: key => key in gl.inputState.state && gl.inputState.state[key],
    };
    initEvents();

    // Set initial values of uniforms
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
        uniform mat4 uModelMatrix;
        uniform mat4 uCameraMatrix;
        uniform mat4 uProjectionMatrix;

        in vec4 aPosition;
        in vec3 aNormal;
        in vec3 aColor;

        out vec3 vNormalVector;
        out vec3 vLightVector;
        out vec3 vEyeVector;

        flat out vec3 vColor;

        void main() {
            mat4 viewMatrix = inverse(uCameraMatrix);
            mat4 modelViewMatrix = viewMatrix * uModelMatrix;

            vec4 P = modelViewMatrix * aPosition;

            vNormalVector = mat3(modelViewMatrix) * aNormal;
            vec4 light = viewMatrix * uLight;
            vLightVector = (light.w == 1.0) ? (P - light).xyz : light.xyz;
            vEyeVector = -P.xyz; // from position to camera

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
        const float lightIntensity = 4.0;

        // Material constants
        const float materialAmbient = 0.2;
        const float materialDiffuse = 0.4;
        const float materialSpecular = 0.6;
        const float materialShininess = 10.0;
        
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
            float attenuation = 1.0 / (lightConstantA * d*d + lightConstantB * d + lightConstantC);

            // compute lighting
            float A = materialAmbient;
            float D = materialDiffuse * diffuse * attenuation;
            float S = materialSpecular * specular * attenuation;

            fragColor.rgb = (((A + D) * vColor) + S) * lightColor * lightIntensity;
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
    program.uModelMatrix = gl.getUniformLocation(program, 'uModelMatrix');
    program.uProjectionMatrix = gl.getUniformLocation(program, 'uProjectionMatrix');
    program.uCameraMatrix = gl.getUniformLocation(program, 'uCameraMatrix');
    program.uLight = gl.getUniformLocation(program, 'uLight');

    return program;
}

/**
 * Initialize the data buffers.
 */
function initBuffers() {
    gl.world = createObject("world");

    // Create the camera
    let camera = createObject("camera");
    {
        const translation = mat4.fromTranslation(mat4.create(), [0, 0, -1]);
        
        const t = camera.localTransform;
        mat4.multiply(t, t, translation);
        //  mat4.multiply(t, t, angleAxisToMat4(0, [0, 1, 0]));
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

        gl.drone = createObject("model");
        gl.drone.model = loadModel(cube_coords, cube_colors, cube_indices);

        // compute drone's transform
        const t = gl.drone.localTransform;
        scaleByConstant(t, 0.2);
    }

    // Generate and load terrain into GPU
    {
        gl.terrain = createObject("model");

        const seed = 782378;
        const terrain = generate_terrain(7, 0.01, Math.seedrandom(seed));
        let [terrainVertices, terrainIndices] = generate_mesh(terrain);
        gl.terrain.model = loadModel(terrainVertices, null, terrainIndices, true);

        // compute terrain's transform
        const t = gl.terrain.localTransform;
        mat4.multiply(t, t, mat4.fromScaling(mat4.create(), [20, 10, 20]));
        mat4.multiply(t, t, angleAxisToMat4(180, [0, 0, 1]));
    }

    gl.world.camera = camera;

    gl.drone.addChild(camera);
    gl.world.addChild(gl.drone);
    gl.world.addChild(gl.terrain);
}

function createObject(type) {
    let obj = {
        type: type,
        localTransform: mat4.identity(mat4.create()),
        parent: null,
        children: []
    };

    obj.addChild = function addChild(child) {
        child.parent = obj;
        obj.children.push(child);
    }

    Object.defineProperty(obj, "transform", {
        get: () => {
            if (obj.parent === null) {
                return obj.localTransform;
            } else {
                return mat4.multiply(mat4.create(), obj.parent.transform, obj.localTransform);
            }
        }
    });

    return obj;
}

/**
 * Initialize event handlers
 */
function initEvents() {
    window.addEventListener('resize', onWindowResize);

    window.addEventListener('keydown', function (e) {
        e.preventDefault()

        gl.inputState.state[e.key] = true;
    });

    window.addEventListener('keyup', function (e) {
        e.preventDefault()

        gl.inputState.state[e.key] = false;
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

    const mv = mat4.perspective(mat4.create(), degrees2radians(90), w / h, 0.0001, 1000);

    // for debugging
    // const mv = mat4.ortho(mat4.create(), -2, 2, -2, 2, 10, -10);

    gl.uniformMatrix4fv(gl.program.uProjectionMatrix, false, mv);
}

function runFrame() {
    updateDroneTransform();

    render();

    window.requestAnimationFrame(runFrame);
}

/**
 * Render the scene.
 */
function render() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.uniformMatrix4fv(gl.program.uCameraMatrix, false, gl.world.camera.transform);
    gl.uniform4fv(gl.program.uLight, [-1, -1, -1, 0]);

    const draw = function (obj) {
        if (obj.type === "model") {
            gl.uniformMatrix4fv(gl.program.uModelMatrix, false, obj.transform);

            gl.bindVertexArray(obj.model.vao);
            gl.drawElements(obj.model.mode, obj.model.count, gl.UNSIGNED_SHORT, 0);
        }

        for (const child of obj.children) {
            draw(child);
        }
    }

    draw(gl.world);

    gl.bindVertexArray(null);
}

function updateDroneTransform() {
    const local = gl.drone.localTransform;

    // !== for booleans does exclusive or

    // moves forward and backward
    if (gl.inputState.isKeyDown("ArrowUp") !== gl.inputState.isKeyDown("ArrowDown")) {
        const factor = gl.inputState.isKeyDown("ArrowUp") ? -1 : 1;
        const t = mat4.fromTranslation(mat4.create(), [0, 0, 0.2 * factor]);
        mat4.multiply(local, local, t);
    }

    const rotations = [];

    // yaw (rotate y-axis)
    if (gl.inputState.isKeyDown("ArrowLeft") !== gl.inputState.isKeyDown("ArrowRight")) {
        const factor = gl.inputState.isKeyDown("ArrowRight") ? 1 : -1;
        rotations.push(angleAxisToQuat(1 * factor, [0, 1, 0]));
    }

    // pitch (rotate x-axis)
    if (gl.inputState.isKeyDown("w") !== gl.inputState.isKeyDown("s")) {
        const factor = gl.inputState.isKeyDown("s") ? 1 : -1;
        rotations.push(angleAxisToQuat(1 * factor, [1, 0, 0]));
    }

    // roll (rotate z-axis)
    if (gl.inputState.isKeyDown("a") !== gl.inputState.isKeyDown("d")) {
        const factor = gl.inputState.isKeyDown("d") ? 1 : -1;
        rotations.push(angleAxisToQuat(1 * factor, [0, 0, 1]));
    }

    let finalRotation = rotations.reduce((a, b) => quat.multiply(a, a, b),
        quat.identity(quat.create()));
    finalRotation = mat4.fromQuat(mat4.create(), finalRotation);

    mat4.multiply(local, local, finalRotation);
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
    const normals = calc_normals(coords, indices, useStrips)
    loadArrayBuffer(normals, gl.program.aNormal, 3, gl.FLOAT);

    if (colors === null) {
        colors = Array(coords.length / 3).fill(null).flatMap(() => [0, 1, 0]);
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
    const transform = mat4.identity(mat4.create());

    const object = { vao, count, mode, transform, children: [] };

    return object
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
 * axis: an array of 3 values.
 */
function angleAxisToQuat(angle, axis) {
    angle = degrees2radians(angle);
    
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
function degrees2radians(angle) {
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

/**
 * Calculates the normals for the vertices given an array of vertices and array of indices to look
 * up into. The triangles are full triangles and not triangle strips.
 *
 * Arguments:
 *    coords - a Float32Array with 3 values per vertex
 *    indices - a regular or typed array
 *    is_tri_strip - defaults to true which means the indices represent a triangle strip
 * Returns:
 *    Float32Array of the normals with 3 values per vertex
 */
function calc_normals(coords, indices, is_tri_strip) {
    if (is_tri_strip !== true && is_tri_strip !== false) { is_tri_strip = true; }
    
    // Start with all vertex normals as <0,0,0>
    let normals = new Float32Array(coords.length);

    // Get temporary variables
    let [N_face, V, U] = [vec3.create(), vec3.create(), vec3.create()];

    // Calculate the face normals for each triangle then add them to the vertices
    let inc = is_tri_strip ? 1 : 3; // triangle strips only go up by 1 index per triangle
    for (let i = 0; i < indices.length - 2; i += inc) {
        // Get the indices of the triangle and then get pointers its coords and normals
        let j = indices[i]*3, k = indices[i+1]*3, l = indices[i+2]*3;
        let A = coords.subarray(j, j+3), B = coords.subarray(k, k+3), C = coords.subarray(l, l+3);
        let NA = normals.subarray(j, j+3), NB = normals.subarray(k, k+3), NC = normals.subarray(l, l+3);

        // Compute normal for the A, B, C triangle and save to N_face (will need to use V and U as temporaries as well)
        vec3.cross(N_face, vec3.subtract(V, B, A), vec3.subtract(U, C, A));
        if (is_tri_strip && (i%2) !== 0) { // every other triangle in a strip is actually reversed
            vec3.negate(N_face, N_face);
        }

        // Add N_face to the 3 normals of the triangle: NA, NB, and NC
        vec3.add(NA, N_face, NA); // NA += N_face
        vec3.add(NB, N_face, NB);
        vec3.add(NC, N_face, NC);
    }

    // Normalize the normals
    for (let i = 0; i < normals.length; i+=3) {
        let N = normals.subarray(i, i+3);
        vec3.normalize(N, N);
    }

    // Return the computed normals
    return normals;
}
