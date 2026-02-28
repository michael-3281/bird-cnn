const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const resultsSection = document.getElementById('results-section');
const heatmapImg = document.getElementById('heatmap-view');
const layerNumDisplay = document.getElementById('layer-num');
const originalPreview = document.getElementById('original-preview');
const resetBtn = document.getElementById('reset-btn');

let heatmapInterval;
let modelWeights = null;
const CLASS_NAMES = ["American Robin", "Bald Eagle", "Blue Jay"];

dropzone.addEventListener('click', () => fileInput.click());

// load weights on startup
async function loadModel() {
    try {
        const response = await fetch('model/model_weights.json');
        modelWeights = await response.json();
        console.log("Model weights loaded locally")
    } catch (e) {
        console.error("Could not load weights:", e);
    }
}
loadModel();

fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file || !modelWeights) return;

    // show loading
    document.getElementById('loading-overlay').classList.remove('hidden');
    dropzone.classList.add('hidden');

    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = async () => {
            // update preview
            originalPreview.innerHTML = `<img src="${img.src}" style="max-width:100%; border-radius:8px;">`;

            // extract pixels
            const canvas = document.createElement('canvas');
            canvas.width = 96;
            canvas.height = 96;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, 96, 96);
            const imageData = ctx.getImageData(0, 0, 96, 96).data;

            // run math
            const results = await runInference(imageData);
            
            document.getElementById('loading-overlay').classList.add('hidden');
            resultsSection.classList.remove('hidden');
            uploadSection.classList.add('hidden');

            updateUIWithResults(results);
            startHeatmapLoop();
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});


function startHeatmapLoop() {
    let layerIdx = 0;
    if (heatmapInterval) clearInterval(heatmapInterval);
    
    heatmapInterval = setInterval(() => {
        if (dynamicHeatmaps.length > 0) {
            heatmapImg.src = dynamicHeatmaps[layerIdx];
            layerNumDisplay.innerText = layerIdx + 1;
            layerIdx = (layerIdx + 1) % dynamicHeatmaps.length;
        }
    }, 1000);
}

function resetApp() {
    clearInterval(heatmapInterval);
    resultsSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    dropzone.classList.remove('hidden');
    fileInput.value = ""; // clear file
    originalPreview.innerHTML = "Original";
}

if(resetBtn) resetBtn.addEventListener('click', resetApp);

// helper math
const CNN = {
    im2col: (input, filterH, filterW, padding = 1, stride = 1) => {
        // get dimensions
        const channels = input.length;
        const height = input[0].length;
        const width = input[0][0].length;
        
        const outH = Math.floor((height + 2 * padding - filterH) / stride) + 1;
        const outW = Math.floor((width + 2 * padding - filterW) / stride) + 1;

        const p = padding;
        const paddedInput = CNN.padInput(input, channels, height, width, p);

        let cols = [];
        for (let c = 0; c < channels; c++) {
            for (let ky = 0; ky < filterH; ky++) {
                for (let kx = 0; kx < filterW; kx++) {
                    let row = [];
                    for (let y = 0; y < outH; y++) {
                        for (let x = 0; x < outW; x++) {
                            row.push(paddedInput[c][y * stride + ky][x * stride + kx]);
                        }
                    }
                    cols.push(row);
                }
            }
        }
        return cols;
    },

    padInput: (img, c, h, w, p) => {
        let padded = Array.from({ length: c }, () => 
            Array.from({ length: h + 2 * p }, () => new Float32Array(w + 2 * p).fill(0))
        );
        for (let i = 0; i < c; i++) {
            for (let j = 0; j < h; j++) {
                for (let k = 0; k < w; k++) {
                    padded[i][j + p][k + p] = img[i][j][k];
                }
            }
        }
        return padded;
    },

    relu: (x) => x.map(v => Math.max(0, v)),

    // softmax
    softmax: (arr) => {
        const max = Math.max(...arr);
        const exps = arr.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b);
        return exps.map(v => v / sum);
    },

    // flattening and matrix multiply
    forwardSoftmax: (input, weights, biases) => {
        return biases.map((b, i) => {
            return input.reduce((acc, val, j) => {
                return acc + (val * weights[j][i]);
            }, 0) + b; // three return statements! wow!
        });
    },

    // i'm sorry...
    convolve: (input, weights, biases, stride = 1, padding = 1) => {
        const [outChannels, inChannels, filterH, filterW] = [weights.length, weights[0].length, weights[0][0].length, weights[0][0][0].length];
        const [_, inH, inW] = [input.length, input[0].length, input[0][0].length];

        // transform input into columns
        const cols = CNN.im2col(input, filterH, filterW, padding, stride);

        // flatten weights to matrix [outChannels, inChannels * filterH * filterW]
        const weightMat = weights.map(filter => filter.flat(2));

        // matrix multiply (weights * cols + biases)   scary!
        const outH = Math.floor((inH + 2 * padding - filterH) / stride) + 1;
        const outW = Math.floor((inW + 2 * padding - filterW) / stride) + 1;

        let res = [];
        for (let i = 0; i < outChannels; i++) {
            let layer = new Float32Array(outH * outW);
            for (let j = 0; j < outH * outW; j++) {
                let sum = biases[i];
                for (let k = 0; k < weightMat[i].length; k++) {
                    sum += weightMat[i][k] * cols[k][j];
                }
                layer[j] = Math.max(0, sum); // apply relu
            }
            // reshape 1d back to 2d [outh, outw]
            let reshapedLayer = [];
            for (let r = 0; r < outH; r++) {
                reshapedLayer.push(layer.slice(r * outW, (r + 1) * outW));
            }
            res.push(reshapedLayer);
        }
        return res;
    },
    // glad that's over! let's not do that again...

    maxPool: (input, size = 2, stride = 2) => {
        const channels = input.length;
        const inH = input[0].length;
        const inW = input[0][0].length;
        const outH = inH / stride;
        const outW = inW / stride;

        let output = [];
        for (let c = 0; c < channels; c++) { // don't worry, i won't make the joke again                                          ->                                                                  (it's like the language)
            let poolLayer = Array.from({ length: outH }, () => new Float32Array(outW));
            for (let i = 0; i < outH; i++) {
                for (let j = 0; j < outW; j++) {
                    let max = -Infinity;
                    for (let py = 0; py < size; py++) {
                        for (let px = 0; px < size; px++) {
                            let val = input[c][i * stride + py][j * stride + px];
                            if (val > max) max = val;
                        }
                    }
                    poolLayer[i][j] = max;
                }
            }
            output.push(poolLayer);
        }
        return output;
    }
};

let dynamicHeatmaps = [];

async function runInference(pixelData) {
    // preprocess
    let x = preprocessPixels(pixelData);
    dynamicHeatmaps = []; // reset

    const convWeights = modelWeights.conv;
    let currentOut = x;

    for (let i = 0; i < 4; i++) {
        // forward (conv + relu)
        currentOut = CNN.convolve(
            currentOut, 
            convWeights[i].weights, 
            convWeights[i].biases
        );

        // save heatmap
        dynamicHeatmaps.push(generateHeatmap(currentOut));

        // forward (maxpool)
        currentOut = CNN.maxPool(currentOut, 2, 2);
    }

    // unpacking and flattening
    const flatOut = [];
    for (let c = 0; c < currentOut.length; c++) { // i'm serious this time, i will not tell another one of my bad jokes
        for (let y = 0; y < currentOut[c].length; y++) {
            flatOut.push(...currentOut[c][y]);
        }
    }

    // softmax layer
    const logits = CNN.forwardSoftmax(
        flatOut,
        modelWeights.softmax.weights,
        modelWeights.softmax.biases
    );
    
    const probs = CNN.softmax(logits);

    // format results
    return CLASS_NAMES.map((name, i) => ({
        name: name,
        confidence: probs[i] * 100
    })).sort((a, b) => b.confidence - a.confidence);
}

// helper to handle canvas-to-array conversion
function preprocessPixels(pixelData) {
    const [h, w] = [96, 96];
    let r = [], g = [], b = [];
    for (let i = 0; i < pixelData.length; i += 4) {
        r.push(pixelData[i] / 255.0);
        g.push(pixelData[i + 1] / 255.0);
        b.push(pixelData[i + 2] / 255.0);
    }
    // reshape to [3, 96, 96]
    return [
        reshape(r, h, w),
        reshape(g, h, w),
        reshape(b, h, w)
    ];
}

function reshape(arr, h, w) {
    let res = [];
    for (let i = 0; i < h; i++) res.push(arr.slice(i * w, (i + 1) * w)); // this should probably be on a new line, but it isn't because i'm evil
    return res;
}

function updateUIWithResults(predictions) {
    const top = predictions[0];
    
    // determine color based on accuracy
    let statusClass = 'status-low';
    if (top.confidence > 80) statusClass = 'status-high';
    else if (top.confidence > 50) statusClass = 'status-med';

    // update prediction
    const topNameEl = document.getElementById('top-prediction');
    const topConfEl = document.querySelector('.confidence-val');
    const topFillEl = document.querySelector('.progress-bar .fill');

    topNameEl.innerText = top.name;
    topConfEl.innerText = `${top.confidence.toFixed(1)}%`;
    
    // apply color
    topFillEl.className = 'fill ' + statusClass;
    topConfEl.style.color = `var(--${statusClass.replace('status-', 'color-')})`;

    // update alts
    const altContainer = document.querySelector('.alternative-predictions');
    altContainer.innerHTML = '<h5>Alternative Predictions</h5>'; 

    predictions.slice(1).forEach(pred => {
        let altStatus = pred.confidence > 80 ? 'status-high' : (pred.confidence > 50 ? 'status-med' : 'status-low');
        const row = document.createElement('div');
        row.className = 'alt-row';
        row.innerHTML = `
            <span>${pred.name}</span>
            <div class="mini-bar"><div class="fill ${altStatus}" style="width: ${pred.confidence}%"></div></div>
            <span>${pred.confidence.toFixed(1)}%</span>
        `;
        altContainer.appendChild(row);
    });
}

function generateHeatmap(layerOutput) {
    const channels = layerOutput.length;
    const h = layerOutput[0].length;
    const w = layerOutput[0][0].length;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);

    let heatmap = new Float32Array(h * w).fill(0);
    for (let c = 0; c < channels; c++) {
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                heatmap[y * w + x] += layerOutput[c][y][x];
            }
        }
    }

    const maxVal = Math.max(...heatmap) || 1;
    for (let i = 0; i < heatmap.length; i++) {
        const val = (heatmap[i] / maxVal) * 255;
        const idx = i * 4;
        imgData.data[idx] = val;           
        imgData.data[idx + 1] = val * 0.8; 
        imgData.data[idx + 2] = 255 - val; 
        imgData.data[idx + 3] = 255;       
    }

    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL();
}

function renderMath() {
    // formula for convolution (it hurts my brain too)
    katex.render("(I * K)(i, j) = \\sum_{m} \\sum_{n} I(i+m, j+n)K(m, n) + b", 
        document.getElementById('math-conv'), { throwOnError: false });

    // formula for relu
    katex.render("f(x) = \\max(0, x)", 
        document.getElementById('math-relu'), { throwOnError: false, displayMode: true });

    // formula for softmax
    katex.render("\\sigma(\\mathbf{z})_i = \\frac{e^{z_i}}{\\sum_{j=1}^K e^{z_j}}", 
        document.getElementById('math-softmax'), { throwOnError: false, displayMode: true });
}

window.addEventListener('DOMContentLoaded', renderMath);