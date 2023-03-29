// Code for generating terrain-like height maps
/* exported generate_terrain generate_mesh */

/**
 * Generates random terrain using the square-diamond algorithm and median filtering.
 * detail determines the size of the returned 2D data (maximum of 7 if using indexed arrays)
 * roughness describes how rough the terrain is (likely to be between 0 and 0.01)
 * The returned data is a 2D array (array of arrays) that represents a height-map (i.e. each
 * value is the distance up the ground is, although the distance can be negative).
 */
function generate_terrain(detail, roughness) {
    // The actual size of the data must be a power of two in each direction
    let size = Math.pow(2, detail) + 1;
    let max = size - 1;
    let map = new Array(size);
    for (let i = 0; i < size; i++) { map[i] = new Float32Array(size); }

    // Start with all random values in the map
    let scale = roughness*size;
    map[0][0] = Math.random()*scale*2 - scale;
    map[max][0] = Math.random()*scale*2 - scale;
    map[max][max] = Math.random()*scale*2 - scale;
    map[0][max] = Math.random()*scale*2 - scale;

    // Recursively run square-diamond algorithm
    divide(map, max);
    
    // Apply median 3x3 filter to the map to smooth it a bit
    map = median3Filter(map);
    
    // Done!
    return map;

    function divide(map, sz) {
        // Recursively divides the map, applying the square-diamond algorithm
        let half = sz / 2, scl = roughness * sz;
        if (half < 1) { return; }

        for (let y = half; y < max; y += sz) {
            for (let x = half; x < max; x += sz) {
                let offset = Math.random() * scl * 2 - scl;
                map[x][y] = offset + square(x, y, half);
            }
        }

        for (let y = 0; y <= max; y += half) {
            for (let x = (y + half) % sz; x <= max; x += sz) {
                let offset = Math.random() * scl * 2 - scl;
                map[x][y] = offset + diamond(x, y, half);
            }
        }

        divide(map, half);
    }

    function average() {
        // Calculates average of all arguments given (except undefined values)
        // This ignores any x/y coordinate outside the map
        let args = Array.prototype.slice.call(arguments).filter(function (n) { return n === 0 || n; });
        let sum = 0;
        for (let i = 0; i < args.length; i++) { sum += args[i]; }
        return sum / args.length;
    }
    
    function median() {
        // Calculates the median of all arguments given (except undefined values)
        let args = Array.prototype.slice.call(arguments).filter(function (n) { return n === 0 || n; });
        args.sort()
        if ((args.length % 2) === 1) { return args[(args.length-1)/2]; }
        return (args[args.length/2-1] + args[args.length/2]) / 2;
    }

    function square(x, y, sz) {
        // Performs a single square computation of the algorithm
        if (x < sz) { return average(map[x+sz][y-sz], map[x+sz][y+sz]); }
        if (x > max - sz) { return average(map[x+sz][y-sz], map[x+sz][y+sz]); }
        return average(map[x-sz][y-sz], map[x+sz][y-sz], map[x+sz][y+sz], map[x-sz][y+sz]);
    }

    function diamond(x, y, sz) {
        // Performs a single computation step of the algorithm
        if (x < sz) { return average(map[x][y-sz], map[x+sz][y], map[x][y+sz]); }
        if (x > max-sz) { return average(map[x][y-sz], map[x][y+sz], map[x-sz][y]); }
        return average(map[x][y-sz], map[x+sz][y], map[x][y+sz], map[x-sz][y]);
    }

    function median3Filter(src) {
        // Applies a 3x3 median filter to the given array-of-arrays.
        let N = src.length, n = N - 1;
        let block = new Float32Array(3*3);
        let dst = new Array(N);
        for (let y = 0; y < N; y++) { dst[y] = new Float32Array(N); }
        // Core of the 'image'
        for (let y = 0; y < N-2; y++) {
            for (let x = 0; x < N-2; x++) {
                for (let cy = 0; cy < 3; cy++) {
                    for (let cx = 0; cx < 3; cx++) {
                        block[cy*3+cx] = src[y+cy][x+cx];
                    }
                }
                block.sort();
                dst[y+1][x+1] = block[4];
            }
        }
        // Corners
        dst[0][0] = median(src[0][0], src[1][0], src[0][1], src[1][1]);
        dst[n][0] = median(src[n][0], src[n][1], src[n-1][0], src[n-1][1]);
        dst[0][n] = median(src[0][n], src[1][n], src[0][n-1], src[1][n-1]);
        dst[n][n] = median(src[n][n], src[n][n-1], src[n-1][n], src[n-1][n-1]);
        // Edges
        for (let y = 1; y < n; y++) {
            dst[y][0] = median(src[y-1][0], src[y][0], src[y+1][0], src[y-1][1], src[y][1], src[y+1][1]);
            dst[y][n] = median(src[y-1][n], src[y][n], src[y+1][n], src[y-1][n-1], src[y][n-1], src[y+1][n-1]);
        }
        for (let x = 1; x < n; x++) {
            dst[0][x] = median(src[0][x-1], src[0][x], src[0][x+1], src[1][x-1], src[1][x], src[1][x+1]);
            dst[n][x] = median(src[n][x-1], src[n][x], src[n][x+1], src[n-1][x-1], src[n-1][x], src[n-1][x+1]);
        }
        // Done
        return dst;
    }
}


/**
 * Generate a mesh of triangles from a 2D dataset that indicates the height of the mesh at every
 * point. The result is to be drawn with gl.drawElements(gl.TRIANGLE_STRIP).
 * 
 * Arguments:
 *     data - 2D input data of heights
 * Returns:
 *     coords - Float32Array of coordinates, each 3 elements
 *     inds   - Uint16Array of indices into coordinates
 */
function generate_mesh(data) {
    let [n, m] = [data.length, data[0].length];

    // Create the coordinates
    let coords = new Float32Array(3*n*m), off = 0;
    for (let i = 0; i < n; i++) {
        let x_val = 2*i/(n-1)-1;
        for (let j = 0; j < m; j++) {
            coords[off++] = x_val;
            coords[off++] = data[i][j];
            coords[off++] = 2*j/(m-1)-1;
        }
    }

    // Create the indices by making rectangles between the grid points
    let indices = new Uint16Array((n-1)*(m+1)*2); off = 0;
    for (let i = 0; i < n-1; i++) {
        let row = i*m;
        for (let j = 0; j < m; j++) {
            indices[off++] = row+j;
            indices[off++] = row+j+m;
        }
        indices[off++] = row+2*m-1;
        indices[off++] = row+m;
    }
    return [coords, indices];
}
