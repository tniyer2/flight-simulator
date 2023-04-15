// Various useful functions
/* exported line_seg_triangle_intersection calc_normals */

const vec3 = glMatrix.vec3;
const _temps = [vec3.create(), vec3.create(), vec3.create(), vec3.create(), vec3.create(), vec3.create(), vec3.create()];

/**
 * Finds the intersection between a line segment and a triangle. The line segment is given by a
 * point (p) and vector (vec). The triangle is given by three points (abc). If there is no
 * intersection, the line segment is parallel to the plane of the triangle, or the triangle is
 * degenerate then null is returned. Otherwise a vec4 is returned that contains the intersection.
 *
 * Each argument must be a vec3 (i.e. 3 element array).
 */
function line_seg_triangle_intersection(p, vec, a, b, c) {
    let [u, v] = [vec3.subtract(_temps[0], b, a), vec3.subtract(_temps[1], c, a)]; // triangle edge vectors
    let uu = vec3.dot(u, u), vv = vec3.dot(v, v), uv = vec3.dot(u, v);
    let tri_scale = uv*uv-uu*vv;
    if (tri_scale === 0) { return null; } // triangle is degenerate
    let n = vec3.cross(_temps[2], u, v); // normal vector of the triangle

    // Find the point where the line intersects the plane of the triangle
    let denom = vec3.dot(n, vec);
    if (denom === 0) { return null; } // line segment is parallel to the plane of the triangle
    let rI = vec3.dot(n, vec3.subtract(_temps[3], a, p)) / denom;
    if (rI < 0 || rI > 1) { return null; } // line segment does not intersect the plane of the triangle
    p = vec3.add(_temps[4], p, vec3.scale(_temps[5], vec, rI)); // the point where the line segment intersects the plane of the triangle

    // Check if the point of intersection lies within the triangle
    let w = vec3.subtract(_temps[6], p, a), wv = vec3.dot(w, v), wu = vec3.dot(w, u);
    let sI = (uv*wv-vv*wu)/tri_scale, tI = (uv*wu-uu*wv)/tri_scale;
    if (sI < 0 || tI < 0 || sI + tI > 1) { return null; } // intersection point is outside of the triangle

    // Return the intersection
    return p;
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
