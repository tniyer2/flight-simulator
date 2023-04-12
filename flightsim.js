// Basic Flight Simulator
// Authors: Bryan Cohen & Tanishq Iyer
'use strict';

// Allow use of glMatrix values directly instead of needing the glMatrix prefix
const vec3 = glMatrix.vec3;
const vec4 = glMatrix.vec4;
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
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.5, 0, 1, 1);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);

    // Initialize the WebGL program and data
    gl.program = initProgram();
    initEvents();
    initBuffers();
    initUniforms();

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
        uniform mat4 uViewMatrix;
        uniform mat4 uProjectionMatrix;

        in vec4 aPosition;
        in vec3 aNormal;
        // in vec3 aColor;

        out vec3 vNormalVector;
        out vec3 vLightVector;
        out vec3 vEyeVector;

        flat out vec3 vColor;

        void main() {
            // mat4 modelViewMatrix = uViewMatrix * uModelMatrix;

            vec4 P = uViewMatrix * aPosition;

            vNormalVector = mat3(uViewMatrix) * aNormal;
            vec4 light = uViewMatrix * uLight;
            vLightVector = (light.w == 1.0) ? (P - light).xyz : light.xyz;
            vEyeVector = -P.xyz; // from position to camera

            gl_Position =  uProjectionMatrix * P;
            vColor = vec3(0.0, 1.0, 0.0); // hardcoded color
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
        const float lightConstantA = 0.3;
        const float lightConstantB = 0.3;
        const float lightConstantC = 0.3;

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

            fragColor.rgb = (((A + D) * vColor) + S) * lightColor * lightIntensity;
            fragColor.a = 1.0;
        }`
    );

    // Link the shaders into a program and use them with the WebGL context
    let program = linkProgram(gl, vert_shader, frag_shader);
    gl.useProgram(program);
    
    // Get the attribute indices
    program.aPosition = gl.getAttribLocation(program, 'aPosition');
    // program.aColor = gl.getAttribLocation(program, 'aColor');
    program.aNormal = gl.getAttribLocation(program, 'aNormal');

    // Get the uniform indices
    program.uModelMatrix = gl.getUniformLocation(program, 'uModelMatrix');
    program.uViewMatrix = gl.getUniformLocation(program, 'uViewMatrix');
    program.uProjectionMatrix = gl.getUniformLocation(program, 'uProjectionMatrix');
    program.uLight = gl.getUniformLocation(program, 'uLight');

    return program;
}

/**
 * Initialize event handlers
 */
function initEvents() {
    window.addEventListener('resize', onWindowResize);
}

function initBuffers() {
    const terrain = generate_mesh(generate_terrain(6, 0.001, Math.seedrandom(782378)));
    const terrainModel = {vertices: terrain[0], indices: terrain[1]};

    gl.terrain = loadModel(terrainModel, true);

    let cube_coords = [
        1, 1, 1, // A
        -1, 1, 1, // B
        -1, -1, 1, // C
        1, -1, 1, // D
        1, -1, -1, // E
        -1, -1, -1, // F
        -1, 1, -1, // G
        1, 1, -1, // H
    ];
    let cube_indices = [
        1, 2, 0, 2, 3, 0,
        7, 6, 1, 0, 7, 1,
        1, 6, 2, 6, 5, 2,
        3, 2, 4, 2, 5, 4,
        6, 7, 5, 7, 4, 5,
        0, 3, 7, 3, 4, 7,
    ];
    gl.cube = loadModel({vertices: cube_coords, indices: cube_indices});
}

function initUniforms() {
    gl.uniform4fv(gl.program.uLight, [0, 3, 0, 1]);
}

/**
 * Load a model into the GPU and return its information.
 */
function loadModel(model, isTriStrip) {
    if (typeof isTriStrip === "undefined") {
        isTriStrip = false;
    }

    // Create and bind the VAO
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    // Load vertex coordinates into GPU
    const coords = Float32Array.from(model.vertices);
    loadArrayBuffer(coords, gl.program.aPosition, 3, gl.FLOAT);
    
    // Load vertex normals into GPU
    const normals = calc_normals(coords, model.indices, isTriStrip);
    loadArrayBuffer(normals, gl.program.aNormal, 3, gl.FLOAT);
    
    // console.log(gl.program.aColor);

    // Load vertex colors into GPU
    /*
    {
        let colors;
        if (typeof defaultColor === "undefined") {
            colors = Float32Array.from(model.colors);
            loadArrayBuffer(colors, gl.program.aColor, 3, gl.Float);
        } else {
            colors = Array(coords.length / 3).fill(null).flatMap(() => defaultColor);
        }
        loadArrayBuffer(colors, gl.program.aColor, 3, gl.Float);
    }*/

    // Load index data into the GPU
    const indBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint16Array.from(model.indices), gl.STATIC_DRAW);
    
    // Cleanup
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    const mode = isTriStrip ? gl.TRIANGLE_STRIP : gl.TRIANGLES;

    return {vao, count: model.indices.length, mode};
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
 * Keep the canvas sized to the window.
 */
function onWindowResize() {
    gl.canvas.width = window.innerWidth;
    gl.canvas.height = window.innerHeight;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    /*
    let size = Math.min(window.innerWidth, window.innerHeight);

    gl.canvas.width = gl.canvas.height = size;
    gl.canvas.style.width = gl.canvas.style.height = size + 'px';
    gl.viewport(0, 0, size, size);
    */

    updateProjectionMatrix();
}

/**
 * Render the scene.
 */
function render() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    updateProjectionMatrix();
    updateViewMatrix();

    for (const model of [gl.cube, gl.terrain]) {
        gl.uniformMatrix4fv(gl.program.uModelMatrix, false, mat4.identity(mat4.create()));
        
        gl.bindVertexArray(model.vao);
        gl.drawElements(model.mode, model.count, gl.UNSIGNED_SHORT, 0);
    }

    // Clean up
    gl.bindVertexArray(null);
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

/**
 * Updates the view matrix.
 */
function updateViewMatrix() {
    let view = mat4.create();
    {
        const isTopView = true;
        
        if (isTopView) {
            mat4.fromRotationTranslation(view,
                angleAxisToQuat(90, [1, 0, 0]),
                [0, 0, -6]
            );
        } else {
            mat4.fromTranslation(view, [0, 0, -1]);
        }
    }
    
    gl.uniformMatrix4fv(gl.program.uViewMatrix, false, view);
}

/**
 * Updates the projection transformation matrix.
 */
function updateProjectionMatrix() {
    let w, h = [gl.canvas.width, gl.canvas.height];
    const p  = mat4.perspective(mat4.create(), degrees2radians(90), w / h, 0.0001, 100);

    gl.uniformMatrix4fv(gl.program.uProjectionMatrix, false, p);
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
