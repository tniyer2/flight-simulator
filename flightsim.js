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
    gl.enable(gl.CULL_FACE);
    
    // Initialize the WebGL program and data
    gl.program = initProgram();
    initBuffers();
    initTransforms();
    initEvents();

    // Set initial values of uniforms
    onWindowResize();

    // Render the scene
    render();
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
    gl.world = createDefaultWorld();

    // Create the camera
    let camera = mat4.create();
    {
        const translation = mat4.fromTranslation(mat4.create(), [0, 0.5, 10]);
        const rotation = angleAxisToMat4(-10, [1, 0, 0]);

        mat4.multiply(camera, camera, translation);
        mat4.multiply(camera, camera, rotation);
    }
    gl.world.camera = {transform: camera};

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

        gl.drone = loadModel(cube_coords, cube_colors, cube_indices);
    }

    // Generate and load terrain into GPU
    {
        const seed = 782378;
        const terrain = generate_terrain(7, 0.01, Math.seedrandom(seed));
        let [terrainVertices, terrainIndices] = generate_mesh(terrain);
        gl.terrain = loadModel(terrainVertices, null, terrainIndices, true);
    }
}

function initTransforms() {
    // compute drone's transform
    {
        const t = gl.drone.transform;
        scaleByConstant(t, 1);
    }

    // compute terrain's transform
    {
        const t = gl.terrain.transform;
        mat4.multiply(t, t, mat4.fromScaling(mat4.create(), [20, 10, 20]));
        mat4.multiply(t, t, angleAxisToMat4(180, [0, 0, 1]));
    }
}

/**
 * Initialize event handlers
 */
function initEvents() {
    window.addEventListener('resize', onWindowResize);
    window.addEventListener('keydown', onKeyDown);
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

    const mv = mat4.perspective(mat4.create(), degrees2radians(90), w / h, 0.0001, 100);

    // for debugging
    // const mv = mat4.ortho(mat4.create(), -2, 2, -2, 2, 10, -10);

    gl.uniformMatrix4fv(gl.program.uProjectionMatrix, false, mv);
}

/**
 * When up or down arrows are pressed, change the radius of the circle.
 */
function onKeyDown(e) {
    e.preventDefault()

    const transform = mat4.create();

    /**
     * up/down arrow    - translate z-axis
     * 
     * left/right arrow - rotate around the y-axis
     * W/S key          - rotate around the x-axis
     * A/D key          - rotate around z-axis
     */
    if (e.key === "ArrowUp" || e.key === "ArrowDown" || e.key ==="ArrowLeft" || e.key ==="ArrowRight") {
        // Get the current camera position 

        // moves forward/backward based on the direction it is facing
        if (e.key === "ArrowUp") {
            mat4.fromTranslation(transform, [0, 0, -1]);
            console.log("key up pressed")
        } else if (e.key === "ArrowDown") {
            mat4.fromTranslation(transform, [0, 0, 1]);
            console.log("key down pressed")
        } else if (e.key ==="ArrowLeft") { // rotate on x axis (Yaw)
            mat4.rotateY(transform, degrees2radians(1));
            console.log("key left pressed")
        } else { // e.key === "Arrow Right"
            mat4.rotateY(transform, degrees2radians(-1));
            console.log("key right pressed")
        }
    }

    if (e.key === "a" || e.key === "d" || e.key === "w" || e.key === "s") {
        
        // Pitch Rotate on Y axis
        if (e.key === "w") {               // W
            console.log("key W pressed")
        } else if (e.key === "s") {        // S
            console.log("key S pressed")
        } else if (e.key ==="a") {         // A
            console.log("key A pressed")
        } else { // e.key === "68"          // D
            console.log("key D pressed")
        }
    }

    const cur = gl.world.camera.transform;
    mat4.multiply(cur, cur, transform);
}

/**
 * Render the scene.
 */
function render(ms) {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // The ms value is the number of miliseconds since some arbitrary time in the past
    // If it is not provided (i.e. render is directly called) then this if statement will grab the current time
    if (!ms) { ms = performance.now(); }

    gl.uniformMatrix4fv(gl.program.uCameraMatrix, false, gl.world.camera.transform);
    gl.uniform4fv(gl.program.uLight, [-1, -1, -1, 0]);

    // For each orbiting object, draw the object
    {
        const draw = function (model) {
            gl.uniformMatrix4fv(gl.program.uModelMatrix, false, model.transform);

            gl.bindVertexArray(model.vao);
            gl.drawElements(model.mode, model.count, gl.UNSIGNED_SHORT, 0);
        }

        draw(gl.drone);
        draw(gl.terrain);

        gl.bindVertexArray(null);
    }

    window.requestAnimationFrame(render);
}

function createDefaultWorld() {
    return {
        type: "world", 
        transform: mat4.identity(mat4.create()), 
        children: [],
        camera: null
    };
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
